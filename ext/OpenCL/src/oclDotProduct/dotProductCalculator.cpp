#include <string>
#include <iostream>
#include <vector>

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

}


  void DotProductCalculator::run()
  {
    auto targetDevice = getTargetDevice();
    reportDeviceInfo(targetDevice);
    reportComputationConstants(NUM_ELEMENTS, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE);



    // Allocate and initialize host arrays
    shrLog("Allocate and Init Host Mem...\n");
    std::vector<cl_float4> srcA(GLOBAL_WORK_SIZE);
    std::vector<cl_float4> srcB(GLOBAL_WORK_SIZE);
    std::vector<cl_float> dst(GLOBAL_WORK_SIZE);
    std::vector<cl_float> Golden(NUM_ELEMENTS);
    shrFillArray((float*)srcA.data(), 4 * NUM_ELEMENTS);
    shrFillArray((float*)srcB.data(), 4 * NUM_ELEMENTS);

    gpuContext_ = clCreateContext(nullptr, 1, &targetDevice, nullptr, nullptr, nullptr);
    commandQueue_ = clCreateCommandQueue(gpuContext_, targetDevice, 0, nullptr);

    srcABuffer_ = clCreateBuffer(gpuContext_, CL_MEM_READ_ONLY, sizeof(cl_float4) * GLOBAL_WORK_SIZE, nullptr, nullptr);
    srcBBuffer_ = clCreateBuffer(gpuContext_, CL_MEM_READ_ONLY, sizeof(cl_float4) * GLOBAL_WORK_SIZE, nullptr, nullptr);
    dstBuffer_ = clCreateBuffer(gpuContext_, CL_MEM_WRITE_ONLY, sizeof(cl_float) * GLOBAL_WORK_SIZE, nullptr, nullptr);

    std::cout << "Creating program" << std::endl;
    size_t programSize = strlen(CL_PROGRAM_DOT_PRODUCT);			// Byte size of kernel code
    gpuProgram_ = clCreateProgramWithSource(gpuContext_, 1, &CL_PROGRAM_DOT_PRODUCT, &programSize, nullptr);
    const char* COMPILATION_FLAGS = "-cl-fast-relaxed-math";
    std::cout << "Building program" << std::endl;
    auto feedback = clBuildProgram(gpuProgram_, 0, nullptr, nullptr, nullptr, nullptr);
    if (feedback != CL_SUCCESS)
    {
      oclLogBuildInfo(gpuProgram_, targetDevice);
      return;
    }

    // Create the kernel
    std::cout << "Creating Kernel (DotProduct)..." << std::endl;
    kernel_ = clCreateKernel(gpuProgram_, "DotProduct", &feedback);

    // Set the Argument values
    shrLog("clSetKernelArg 0 - 3...\n\n");
    clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)& srcABuffer_);
    clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)& srcBBuffer_);
    clSetKernelArg(kernel_, 2, sizeof(cl_mem), (void*)& dstBuffer_);
    clSetKernelArg(kernel_, 3, sizeof(cl_int), (void*)& NUM_ELEMENTS);

    // --------------------------------------------------------
    // Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    shrLog("clEnqueueWriteBuffer (SrcA and SrcB)...\n");
    clEnqueueWriteBuffer(commandQueue_, srcABuffer_, CL_FALSE, 0, sizeof(cl_float) * GLOBAL_WORK_SIZE * 4, srcA.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commandQueue_, srcBBuffer_, CL_FALSE, 0, sizeof(cl_float) * GLOBAL_WORK_SIZE * 4, srcB.data(), 0, nullptr, nullptr);

    // Launch kernel
    shrLog("clEnqueueNDRangeKernel (DotProduct)...\n");
    clEnqueueNDRangeKernel(commandQueue_, kernel_, 1, nullptr, &GLOBAL_WORK_SIZE, &LOCAL_WORK_SIZE, 0, nullptr, nullptr);

    // Read back results and check accumulated errors
    shrLog("clEnqueueReadBuffer (Dst)...\n\n");
    clEnqueueReadBuffer(commandQueue_, dstBuffer_, CL_TRUE, 0, sizeof(cl_float) * GLOBAL_WORK_SIZE, dst.data(), 0, nullptr, nullptr);

    // Compute and compare results for golden-host and report errors and pass/fail
    shrLog("Comparing against Host/C++ computation...\n\n");
    DotProductHost((const float*)srcA.data(), (const float*)srcB.data(), (float*)Golden.data(), NUM_ELEMENTS);
    shrBOOL bMatch = shrComparefet((const float*)Golden.data(), (const float*)dst.data(), (unsigned int)NUM_ELEMENTS, 0.0f, 0);
    std::cout << std::boolalpha;
    std::cout << "COMPARING STATUS : " << bMatch << std::endl;

  }

DotProductCalculator::~DotProductCalculator()
{
  if (srcABuffer_) clReleaseMemObject(srcABuffer_);
  if (srcBBuffer_) clReleaseMemObject(srcBBuffer_);
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