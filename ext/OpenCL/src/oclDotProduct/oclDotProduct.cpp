#include <string>
#include <oclUtils.h>
#include <shrQATest.h>
#include <iostream>
#include <vector>
#include "DotProduct.cl"

size_t szParmDataBytes;			// Byte size of context information

// *********************************************************************
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup (int iExitCode);
void (*pCleanup)(int) = &Cleanup;


void reportInt(const cl_device_id deviceId, const cl_device_info deviceInfoConstant, std::string msg)
{
  cl_uint info;    
  clGetDeviceInfo(deviceId, deviceInfoConstant, sizeof(info), &info, nullptr);
  std::cout << msg << info << std::endl;
}

// *********************************************************************
int main(int argc, char** argv)
{
  // Get the NVIDIA platform
  cl_platform_id platformId;      // OpenCL platform
  auto feedback = oclGetPlatformID(&platformId);
  std::cout << "PlatformID = " << platformId << std::endl;
  //Get all the devices
  cl_uint numDevices = 0;           // Number of devices available
  std::cout << "Get the Device info and select Device..." << std::endl;
  clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  std::vector<cl_device_id> devices(numDevices);      // OpenCL device
  clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);

  // Get command line device options and config accordingly
  std::cout << "# of Devices Available = " << numDevices << std::endl;
  const cl_uint TARGET_DEVICE = 0;	        // Default Device to compute on
  std::cout << "Using Device %u: " << TARGET_DEVICE << std::endl;
  oclPrintDevName(LOGBOTH, devices[TARGET_DEVICE]);

  reportInt(devices[TARGET_DEVICE], CL_DEVICE_MAX_COMPUTE_UNITS, "Number of compute units = ");

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

  auto gpuContext = clCreateContext(nullptr, 1, &devices[TARGET_DEVICE], nullptr, nullptr, nullptr);
  auto commandQueue = clCreateCommandQueue(gpuContext, devices[TARGET_DEVICE], 0, nullptr);

  auto srcABuffer = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(cl_float4) * globalWorkSize, nullptr, nullptr);
  auto srcBBuffer = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(cl_float4) * globalWorkSize, nullptr, nullptr);
  auto dstBuffer = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * globalWorkSize, nullptr, nullptr);

  std::cout << "Creating program" << std::endl;
  size_t programSize = strlen(CL_PROGRAM_DOT_PRODUCT);			// Byte size of kernel code
  auto gpuProgram = clCreateProgramWithSource(gpuContext, 1, &CL_PROGRAM_DOT_PRODUCT, &programSize, &feedback);
  const char* COMPILATION_FLAGS = "-cl-fast-relaxed-math";
  std::cout << "Building program" << std::endl;
  feedback = clBuildProgram(gpuProgram, 0, nullptr, nullptr, nullptr, nullptr);
  if (feedback != CL_SUCCESS)
  {
    oclLogBuildInfo(gpuProgram, devices[TARGET_DEVICE]);
    return 1;
  }

  // Create the kernel
  std::cout << "Creating Kernel (DotProduct)..." << std::endl;
  auto kernel = clCreateKernel(gpuProgram, "DotProduct", &feedback);

  // Set the Argument values
  shrLog("clSetKernelArg 0 - 3...\n\n");
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& srcABuffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& srcBBuffer);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& dstBuffer);
  clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)& NUM_ELEMENTS);

  // --------------------------------------------------------
  // Core sequence... copy input data to GPU, compute, copy results back

  // Asynchronous write of data to GPU device
  shrLog("clEnqueueWriteBuffer (SrcA and SrcB)...\n");
  clEnqueueWriteBuffer(commandQueue, srcABuffer, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcA.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(commandQueue, srcBBuffer, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcB.data(), 0, nullptr, nullptr);

  // Launch kernel
  shrLog("clEnqueueNDRangeKernel (DotProduct)...\n");
  feedback = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, &LOCAL_WORK_SIZE, 0, NULL, NULL);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Read back results and check accumulated errors
  shrLog("clEnqueueReadBuffer (Dst)...\n\n");
  feedback = clEnqueueReadBuffer(commandQueue, dstBuffer, CL_TRUE, 0, sizeof(cl_float) * globalWorkSize, dst.data(), 0, NULL, NULL);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Compute and compare results for golden-host and report errors and pass/fail
  shrLog("Comparing against Host/C++ computation...\n\n");
  DotProductHost((const float*)srcA.data(), (const float*)srcB.data(), (float*)Golden.data(), NUM_ELEMENTS);
  shrBOOL bMatch = shrComparefet((const float*)Golden.data(), (const float*)dst.data(), (unsigned int)NUM_ELEMENTS, 0.0f, 0);
  std::cout << std::boolalpha;
  std::cout << "COMPARING STATUS : " << bMatch << std::endl;

  // Cleanup and leave
  if (srcABuffer) clReleaseMemObject(srcABuffer);
  if (srcBBuffer) clReleaseMemObject(srcBBuffer);
  if (dstBuffer) clReleaseMemObject(dstBuffer);
  if (kernel) clReleaseKernel(kernel);
  if (gpuProgram) clReleaseProgram(gpuProgram);
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (gpuContext) clReleaseContext(gpuContext);

  return 0;
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

// Cleanup and exit code
// *********************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("Starting Cleanup...\n\n");


    //if (cdDevices) free(cdDevices);

    //shrQAFinishExit(*gp_argc, (const char **)*gp_argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}
