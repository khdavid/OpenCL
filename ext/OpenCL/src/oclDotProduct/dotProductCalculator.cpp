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
  std::vector<cl_device_id> getDevices()
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
    return devices;
  }
}

void DotProductCalculator::run()
  {
    auto devices = getDevices();
    const cl_uint TARGET_DEVICE = 0;	        // Default Device to compute on
    std::cout << "Using Device %u: " << TARGET_DEVICE << std::endl;
    oclPrintDevName(LOGBOTH, devices[TARGET_DEVICE]);

    reportConstant<cl_uint>(devices[TARGET_DEVICE], CL_DEVICE_MAX_COMPUTE_UNITS, "Number of compute units = ");
    reportConstant<size_t>(devices[TARGET_DEVICE], CL_DEVICE_MAX_WORK_GROUP_SIZE, "Max Number work groups = ");
    reportConstant<size_t>(devices[TARGET_DEVICE], CL_KERNEL_WORK_GROUP_SIZE, "Max Number kernel work groups = ");


    // start logs
    const int NUM_ELEMENTS = 1277944;	    // Length of float arrays to process (odd # for illustration)
    shrLog("Starting...\n\n# of float elements per Array \t= %u\n", NUM_ELEMENTS);

    // set and log Global and Local work size dimensions
    const size_t LOCAL_WORK_SIZE = 256;
    const auto globalWorkSize = shrRoundUp((int)LOCAL_WORK_SIZE, NUM_ELEMENTS);  // rounded up to the nearest multiple of the LocalWorkSize
    shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n",
      globalWorkSize, LOCAL_WORK_SIZE, (globalWorkSize % LOCAL_WORK_SIZE + globalWorkSize / LOCAL_WORK_SIZE));

    // Allocate and initialize host arrays
    shrLog("Allocate and Init Host Mem...\n");
    std::vector<cl_float4> srcA(globalWorkSize);
    std::vector<cl_float4> srcB(globalWorkSize);
    std::vector<cl_float> dst(globalWorkSize);
    std::vector<cl_float> Golden(NUM_ELEMENTS);
    shrFillArray((float*)srcA.data(), 4 * NUM_ELEMENTS);
    shrFillArray((float*)srcB.data(), 4 * NUM_ELEMENTS);

    gpuContext_ = clCreateContext(nullptr, 1, &devices[TARGET_DEVICE], nullptr, nullptr, nullptr);
    commandQueue_ = clCreateCommandQueue(gpuContext_, devices[TARGET_DEVICE], 0, nullptr);

    srcABuffer_ = clCreateBuffer(gpuContext_, CL_MEM_READ_ONLY, sizeof(cl_float4) * globalWorkSize, nullptr, nullptr);
    srcBBuffer_ = clCreateBuffer(gpuContext_, CL_MEM_READ_ONLY, sizeof(cl_float4) * globalWorkSize, nullptr, nullptr);
    dstBuffer_ = clCreateBuffer(gpuContext_, CL_MEM_WRITE_ONLY, sizeof(cl_float) * globalWorkSize, nullptr, nullptr);

    std::cout << "Creating program" << std::endl;
    size_t programSize = strlen(CL_PROGRAM_DOT_PRODUCT);			// Byte size of kernel code
    gpuProgram_ = clCreateProgramWithSource(gpuContext_, 1, &CL_PROGRAM_DOT_PRODUCT, &programSize, nullptr);
    const char* COMPILATION_FLAGS = "-cl-fast-relaxed-math";
    std::cout << "Building program" << std::endl;
    auto feedback = clBuildProgram(gpuProgram_, 0, nullptr, nullptr, nullptr, nullptr);
    if (feedback != CL_SUCCESS)
    {
      oclLogBuildInfo(gpuProgram_, devices[TARGET_DEVICE]);
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
    clEnqueueWriteBuffer(commandQueue_, srcABuffer_, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcA.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commandQueue_, srcBBuffer_, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcB.data(), 0, nullptr, nullptr);

    // Launch kernel
    shrLog("clEnqueueNDRangeKernel (DotProduct)...\n");
    clEnqueueNDRangeKernel(commandQueue_, kernel_, 1, nullptr, &globalWorkSize, &LOCAL_WORK_SIZE, 0, nullptr, nullptr);

    // Read back results and check accumulated errors
    shrLog("clEnqueueReadBuffer (Dst)...\n\n");
    clEnqueueReadBuffer(commandQueue_, dstBuffer_, CL_TRUE, 0, sizeof(cl_float) * globalWorkSize, dst.data(), 0, nullptr, nullptr);

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