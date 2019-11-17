#include <string>
#include <iostream>
#include <vector>
#include <optional>

#include <oclUtils.h>
#include <shrQATest.h>

#include "dotProductCalculator.h"

#include "DotProduct.cl"

size_t szParmDataBytes;			// Byte size of context information

// *********************************************************************
namespace
{
  const size_t NUM_ELEMENTS = 1277944;
  const size_t LOCAL_WORK_SIZE = 256;
  const size_t GLOBAL_WORK_SIZE = shrRoundUp((int)LOCAL_WORK_SIZE, NUM_ELEMENTS);  // rounded up to the nearest multiple of the LocalWorkSize

  template <class T>
  void reportConstant(const cl_device_id deviceId, const cl_device_info deviceInfoConstant, std::string msg)
  {
    T info;
    clGetDeviceInfo(deviceId, deviceInfoConstant, sizeof(info), &info, nullptr);
    std::cout << msg << info << std::endl;
  }

  void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
  {
    int i, j, k;
    for (i = 0, j = 0; i < iNumElements; i++)
    {
      pfResult[i] = 0.0f;
      for (k = 0; k < 4; k++, j++)
      {
        pfResult[i] += pfData1[j] * pfData2[j];
      }
    }
  }
  cl_device_id getTargetDevice()
  {
    // Get the NVIDIA platform
    cl_platform_id platformId;      // OpenCL platform
    oclGetPlatformID(&platformId);
    std::cout << "PlatformID = " << platformId << std::endl;
    //Get all the devices
    cl_uint numDevices = 0;           // Number of devices available
    std::cout << "Get the Device info and select Device..." << std::endl;
    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    std::vector<cl_device_id> devices(numDevices);      // OpenCL device
    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    std::cout << "# of Devices Available = " << numDevices << std::endl;

    const cl_uint TARGET_DEVICE = 0;	        // Default Device to compute on
    std::cout << "Using Device %u: " << TARGET_DEVICE << std::endl;
    oclPrintDevName(LOGBOTH, devices[TARGET_DEVICE]);

    return devices[TARGET_DEVICE];
  }

  void reportDeviceInfo(const cl_device_id targetDevice)
  {
    reportConstant<cl_uint>(targetDevice, CL_DEVICE_MAX_COMPUTE_UNITS, "Number of compute units = ");
    reportConstant<size_t>(targetDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, "Max Number work groups = ");
    reportConstant<size_t>(targetDevice, CL_KERNEL_WORK_GROUP_SIZE, "Max Number kernel work groups = ");
  }

  void reportComputationConstants(size_t numElements, size_t globalWorkSize, size_t localWorkSize)
  {
    // start logs
    shrLog("Starting...\n\n# of float elements per Array \t= %u\n", numElements);
    shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n",
      globalWorkSize, localWorkSize, (globalWorkSize % localWorkSize + globalWorkSize / localWorkSize));
  }

  struct Data
  {
    std::vector<cl_float4> sourceA = std::vector<cl_float4>(GLOBAL_WORK_SIZE);
    std::vector<cl_float4> sourceB = std::vector<cl_float4>(GLOBAL_WORK_SIZE);
    std::vector<cl_float> dotProductResults = std::vector<cl_float>(GLOBAL_WORK_SIZE);
    std::vector<cl_float> dotProductResultsValidation = std::vector<cl_float>(GLOBAL_WORK_SIZE);
  };

  void populateDataInput(Data& data, size_t numElements)
  {
    // Allocate and initialize host arrays
    shrLog("Allocate and Init Host Mem...\n");
    shrFillArray((float*)data.sourceA.data(), int (4 * numElements));
    shrFillArray((float*)data.sourceB.data(), int (4 * numElements));
  }

  std::optional<cl_program> buildProgram(cl_context gpuContext, cl_device_id targetDevice)
  {
    std::cout << "Creating program" << std::endl;
    size_t programSize = strlen(CL_PROGRAM_DOT_PRODUCT);			// Byte size of kernel code
    auto gpuProgram = clCreateProgramWithSource(gpuContext, 1, &CL_PROGRAM_DOT_PRODUCT, &programSize, nullptr);
    //const char* COMPILATION_FLAGS = "-cl-fast-relaxed-math";
    std::cout << "Building program" << std::endl;
    auto feedback = clBuildProgram(gpuProgram, 0, nullptr, nullptr, nullptr, nullptr);
    if (feedback != CL_SUCCESS)
    {
      oclLogBuildInfo(gpuProgram, targetDevice);
      return std::nullopt;
    }

    return gpuProgram;
  }
}


void DotProductCalculator::run()
{
  auto targetDevice = getTargetDevice();
  reportDeviceInfo(targetDevice);
  reportComputationConstants(NUM_ELEMENTS, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE);
  Data data;
  populateDataInput(data, NUM_ELEMENTS);


  gpuContext_ = clCreateContext(nullptr, 1, &targetDevice, nullptr, nullptr, nullptr);
  auto gpuProgramOptional = buildProgram(gpuContext_, targetDevice);

  if (!gpuProgramOptional) 
    return;

  gpuProgram_ = *gpuProgramOptional;;



  sourceABuffer_ = clCreateBuffer(gpuContext_, CL_MEM_READ_ONLY, sizeof(cl_float4) * GLOBAL_WORK_SIZE, nullptr, nullptr);
  sourceBBuffer_ = clCreateBuffer(gpuContext_, CL_MEM_READ_ONLY, sizeof(cl_float4) * GLOBAL_WORK_SIZE, nullptr, nullptr);
  dstBuffer_ = clCreateBuffer(gpuContext_, CL_MEM_WRITE_ONLY, sizeof(cl_float) * GLOBAL_WORK_SIZE, nullptr, nullptr);

  // Create the kernel
  std::cout << "Creating Kernel (DotProduct)..." << std::endl;
  kernel_ = clCreateKernel(gpuProgram_, "DotProduct", nullptr);

  // Set the Argument values
  shrLog("clSetKernelArg 0 - 3...\n\n");
  clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)& sourceABuffer_);
  clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)& sourceBBuffer_);
  clSetKernelArg(kernel_, 2, sizeof(cl_mem), (void*)& dstBuffer_);
  clSetKernelArg(kernel_, 3, sizeof(cl_int), (void*)& NUM_ELEMENTS);

  // --------------------------------------------------------
  // Core sequence... copy input data to GPU, compute, copy results back

  commandQueue_ = clCreateCommandQueue(gpuContext_, targetDevice, 0, nullptr);

  // Asynchronous write of data to GPU device
  shrLog("clEnqueueWriteBuffer (SrcA and SrcB)...\n");
  clEnqueueWriteBuffer(commandQueue_, sourceABuffer_, CL_FALSE, 0, sizeof(cl_float) * GLOBAL_WORK_SIZE * 4, data.sourceA.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(commandQueue_, sourceBBuffer_, CL_FALSE, 0, sizeof(cl_float) * GLOBAL_WORK_SIZE * 4, data.sourceB.data(), 0, nullptr, nullptr);

  // Launch kernel
  shrLog("clEnqueueNDRangeKernel (DotProduct)...\n");
  clEnqueueNDRangeKernel(commandQueue_, kernel_, 1, nullptr, &GLOBAL_WORK_SIZE, &LOCAL_WORK_SIZE, 0, nullptr, nullptr);

  // Read back results and check accumulated errors
  shrLog("clEnqueueReadBuffer (Dst)...\n\n");
  clEnqueueReadBuffer(commandQueue_, dstBuffer_, CL_TRUE, 0, sizeof(cl_float) * GLOBAL_WORK_SIZE, data.dotProductResults.data(), 0, nullptr, nullptr);

  // Compute and compare results for golden-host and report errors and pass/fail
  shrLog("Comparing against Host/C++ computation...\n\n");
  DotProductHost((const float*)data.sourceA.data(), (const float*)data.sourceB.data(), (float*)data.dotProductResultsValidation.data(), NUM_ELEMENTS);
  shrBOOL bMatch = shrComparefet((const float*)data.dotProductResultsValidation.data(), (const float*)data.dotProductResults.data(), (unsigned int)NUM_ELEMENTS, 0.0f, 0);
  std::cout << std::boolalpha;
  std::cout << "COMPARING STATUS : " << bMatch << std::endl;
}

DotProductCalculator::~DotProductCalculator()
{
  if (sourceABuffer_) clReleaseMemObject(sourceABuffer_);
  if (sourceBBuffer_) clReleaseMemObject(sourceBBuffer_);
  if (dstBuffer_) clReleaseMemObject(dstBuffer_);
  if (kernel_) clReleaseKernel(kernel_);
  if (gpuProgram_) clReleaseProgram(gpuProgram_);
  if (commandQueue_) clReleaseCommandQueue(commandQueue_);
  if (gpuContext_) clReleaseContext(gpuContext_);
}


// *********************************************************************
int main(int argc, char** argv)
{
  DotProductCalculator dotProductCalculator;
  dotProductCalculator.run();
  return 0;
}