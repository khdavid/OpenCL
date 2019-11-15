#include <string>
#include <oclUtils.h>
#include <shrQATest.h>
#include <iostream>
#include <vector>

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "DotProduct.cl";

// Host buffers for demo
// *********************************************************************
//void *srcB, *dst;        // Host buffers for OpenCL test
//void* Golden;                   // Host buffer for host golden processing cross check

// OpenCL Vars
//cl_platform_id cpPlatform;      // OpenCL platform
//cl_device_id   *cdDevices;      // OpenCL device
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;               // OpenCL device source buffer B 
cl_mem cmDevDst;                // OpenCL device destination buffer 
//size_t szGlobalWorkSize;        // Total # of work items in the 1D range
//size_t szLocalWorkSize;		    // # of work items in the 1D work group	
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code
//cl_int ciErrNum;			    // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 
//const char* cExecutableName = NULL;

//shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);
void Cleanup (int iExitCode);
void (*pCleanup)(int) = &Cleanup;

int *gp_argc = NULL;
char ***gp_argv = NULL;

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
  std::cout << "clGetPlatformID...\n" << std::endl;
  //Get all the devices
  cl_uint numDevices = 0;           // Number of devices available
  std::cout << "Get the Device info and select Device..." << std::endl;
  feedback = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  std::vector<cl_device_id> devices(numDevices);      // OpenCL device
  clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), NULL);

  // Get command line device options and config accordingly
  std::cout << "# of Devices Available = " << numDevices << std::endl;
  const cl_uint cTargetDevice = 0;	        // Default Device to compute on
  std::cout << "Using Device %u: " << cTargetDevice << std::endl;
  oclPrintDevName(LOGBOTH, devices[cTargetDevice]);

  reportInt(devices[cTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, "Number of compute units = ");

  // start logs
  const int cNumElements = 1277944;	    // Length of float arrays to process (odd # for illustration)
  shrLog("Starting...\n\n# of float elements per Array \t= %u\n", cNumElements);

  // set and log Global and Local work size dimensions
  const size_t localWorkSize = 256;
  const auto globalWorkSize = shrRoundUp((int)localWorkSize, cNumElements);  // rounded up to the nearest multiple of the LocalWorkSize
  shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n",
    globalWorkSize, localWorkSize, (globalWorkSize % localWorkSize + globalWorkSize / localWorkSize));

  // Allocate and initialize host arrays
  shrLog("Allocate and Init Host Mem...\n");
  std::vector<cl_float4> srcA(globalWorkSize);
  std::vector<cl_float4> srcB(globalWorkSize);
  std::vector<cl_float> dst(globalWorkSize);
  std::vector<cl_float> Golden(cNumElements);
  shrFillArray((float*)srcA.data(), 4 * cNumElements);
  shrFillArray((float*)srcB.data(), 4 * cNumElements);

  // Get the NVIDIA platform
  feedback = oclGetPlatformID(&platformId);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Get a GPU device
  feedback = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &devices[cTargetDevice], NULL);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Create the context
  cxGPUContext = clCreateContext(0, 1, &devices[cTargetDevice], NULL, NULL, &feedback);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Create a command-queue
  shrLog("clCreateCommandQueue...\n");
  cqCommandQueue = clCreateCommandQueue(cxGPUContext, devices[cTargetDevice], 0, &feedback);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
  shrLog("clCreateBuffer (SrcA, SrcB and Dst in Device GMEM)...\n");
  cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * globalWorkSize * 4, NULL, &feedback);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);
  cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * globalWorkSize * 4, NULL, &feedback);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);
  cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * globalWorkSize, NULL, &feedback);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Read the OpenCL kernel in from source file
  shrLog("oclLoadProgSource (%s)...\n", cSourceFile);
  cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
  oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
  cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
  oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

  // Create the program
  shrLog("clCreateProgramWithSource...\n");
  cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)& cSourceCL, &szKernelLength, &feedback);

  // Build the program with 'mad' Optimization option
#ifdef MAC
  char* flags = "-cl-fast-relaxed-math -DMAC";
#else
  char* flags = "-cl-fast-relaxed-math";
#endif
  shrLog("clBuildProgram...\n");
  feedback = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
  if (feedback != CL_SUCCESS)
  {
    // write out standard error, Build Log and PTX, then cleanup and exit
    shrLogEx(LOGBOTH | ERRORMSG, feedback, STDERROR);
    oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
    oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclDotProduct.ptx");
    Cleanup(EXIT_FAILURE);
  }

  // Create the kernel
  shrLog("clCreateKernel (DotProduct)...\n");
  ckKernel = clCreateKernel(cpProgram, "DotProduct", &feedback);

  // Set the Argument values
  shrLog("clSetKernelArg 0 - 3...\n\n");
  feedback = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)& cmDevSrcA);
  feedback |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)& cmDevSrcB);
  feedback |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)& cmDevDst);
  feedback |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)& cNumElements);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // --------------------------------------------------------
  // Core sequence... copy input data to GPU, compute, copy results back

  // Asynchronous write of data to GPU device
  shrLog("clEnqueueWriteBuffer (SrcA and SrcB)...\n");
  feedback = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcA.data(), 0, NULL, NULL);
  feedback |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * globalWorkSize * 4, srcB.data(), 0, NULL, NULL);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Launch kernel
  shrLog("clEnqueueNDRangeKernel (DotProduct)...\n");
  feedback = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Read back results and check accumulated errors
  shrLog("clEnqueueReadBuffer (Dst)...\n\n");
  feedback = clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * globalWorkSize, dst.data(), 0, NULL, NULL);
  oclCheckErrorEX(feedback, CL_SUCCESS, pCleanup);

  // Compute and compare results for golden-host and report errors and pass/fail
  shrLog("Comparing against Host/C++ computation...\n\n");
  DotProductHost((const float*)srcA.data(), (const float*)srcB.data(), (float*)Golden.data(), cNumElements);
  shrBOOL bMatch = shrComparefet((const float*)Golden.data(), (const float*)dst.data(), (unsigned int)cNumElements, 0.0f, 0);
  std::cout << std::boolalpha;
  std::cout << "COMPARING STATUS : " << bMatch << std::endl;

  // Cleanup and leave
  Cleanup(EXIT_SUCCESS);
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
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if (cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if (cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if (cmDevDst)clReleaseMemObject(cmDevDst);


    //if (cdDevices) free(cdDevices);

    //shrQAFinishExit(*gp_argc, (const char **)*gp_argv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}
