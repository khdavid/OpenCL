#include <string>
#include <oclUtils.h>
#include <shrQATest.h>
#include <iostream>
#include <vector>
#include "DotProduct.cl"

cl_kernel ckKernel;             // OpenCL kernel
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
  feedback = clBuildProgram(gpuProgram, 0, NULL, NULL, NULL, NULL);
  if (feedback != CL_SUCCESS)
  {
    oclLogBuildInfo(gpuProgram, devices[TARGET_DEVICE]);
    return 1;
  }

  // Create the kernel
  shrLog("clCreateKernel (DotProduct)...\n");
  ckKernel = clCreateKernel(gpuProgram, "DotProduct", &feedback);

  // Set the Argument values
  shrLog("clSetKernelArg 0 - 3...\n\n");
  feedback = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)& srcABuffer);
  feedback |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)& srcBBuffer);
  feedback |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)& dstBuffer);
  feedback |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)& NUM_ELEMENTS);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // --------------------------------------------------------
  // Core sequence... copy input data to GPU, compute, copy results back

  // Asynchronous write of data to GPU device
  shrLog("clEnqueueWriteBuffer (SrcA and SrcB)...\n");
  feedback = clEnqueueWriteBuffer(commandQueue, srcABuffer, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcA.data(), 0, NULL, NULL);
  feedback |= clEnqueueWriteBuffer(commandQueue, srcBBuffer, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcB.data(), 0, NULL, NULL);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Launch kernel
  shrLog("clEnqueueNDRangeKernel (DotProduct)...\n");
  feedback = clEnqueueNDRangeKernel(commandQueue, ckKernel, 1, NULL, &globalWorkSize, &LOCAL_WORK_SIZE, 0, NULL, NULL);
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
  if (gpuProgram) clReleaseProgram(gpuProgram);
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (gpuContext) clReleaseContext(gpuContext);

  return 0;
}






/*
int main2(int argc, char **argv)
{
    gp_argc = &argc;
    gp_argv = &argv;

    shrQAStart(argc, argv);

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("clGetPlatformID...\n"); 

    //Get all the devices
    cl_uint uiNumDevices = 0;           // Number of devices available
    cl_uint uiTargetDevice = 0;	        // Default Device to compute on
    cl_uint uiNumComputeUnits;          // Number of compute units (SM's on NV GPU)
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    // Get command line device options and config accordingly
    shrLog("  # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog("  Using Device %u: ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);
    ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("\n  # of Compute Units = %u\n", uiNumComputeUnits); 

    // get command line arg for quick test, if provided
    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");

    // start logs
	cExecutableName = argv[0];
    shrSetLogFileName ("oclDotProduct.txt");
    shrLog("%s Starting...\n\n# of float elements per Array \t= %u\n", argv[0], iNumElements); 

    // set and log Global and Local work size dimensions
    szLocalWorkSize = 256;
    szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
    shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
           szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

    // Allocate and initialize host arrays
    shrLog( "Allocate and Init Host Mem...\n"); 
    srcA = (void *)malloc(sizeof(cl_float4) * szGlobalWorkSize);
    srcB = (void *)malloc(sizeof(cl_float4) * szGlobalWorkSize);
    dst = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize);
    Golden = (void *)malloc(sizeof(cl_float) * iNumElements);
    shrFillArray((float*)srcA, 4 * iNumElements);
    shrFillArray((float*)srcB, 4 * iNumElements);

    // Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get a GPU device
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevices[uiTargetDevice], NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue
    shrLog("clCreateCommandQueue...\n"); 
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    shrLog("clCreateBuffer (SrcA, SrcB and Dst in Device GMEM)...\n"); 
    cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize * 4, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize * 4, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Read the OpenCL kernel in from source file
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // Create the program
    shrLog("clCreateProgramWithSource...\n"); 
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);

        // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif
    shrLog("clBuildProgram...\n"); 
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclDotProduct.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // Create the kernel
    shrLog("clCreateKernel (DotProduct)...\n"); 
    ckKernel = clCreateKernel(cpProgram, "DotProduct", &ciErrNum);

    // Set the Argument values
    shrLog("clSetKernelArg 0 - 3...\n\n"); 
    ciErrNum = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevDst);
    ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&iNumElements);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // --------------------------------------------------------
    // Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    shrLog("clEnqueueWriteBuffer (SrcA and SrcB)...\n"); 
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize * 4, srcA, 0, NULL, NULL);
    ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize * 4, srcB, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Launch kernel
    shrLog("clEnqueueNDRangeKernel (DotProduct)...\n"); 
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Read back results and check accumulated errors
    shrLog("clEnqueueReadBuffer (Dst)...\n\n"); 
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * szGlobalWorkSize, dst, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Compute and compare results for golden-host and report errors and pass/fail
    shrLog("Comparing against Host/C++ computation...\n\n"); 
    DotProductHost ((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
    shrBOOL bMatch = shrComparefet((const float*)Golden, (const float*)dst, (unsigned int)iNumElements, 0.0f, 0);

    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
    return 0;
}*/

// "Golden" Host processing dot product function for comparison purposes
// *********************************************************************
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
	if(ckKernel)clReleaseKernel(ckKernel);  


    //if (cdDevices) free(cdDevices);

    //shrQAFinishExit(*gp_argc, (const char **)*gp_argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}
