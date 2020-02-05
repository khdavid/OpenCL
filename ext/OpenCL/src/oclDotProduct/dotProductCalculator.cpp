#include <string>
#include <iostream>
#include <vector>
#include <optional>

#include <oclUtils.h>
#include <shrQATest.h>

#include "dotProductCalculator.h"
#include "timer.h"

#include "DotProduct.cl"
#include <future>

size_t szParmDataBytes;			// Byte size of context information

// *********************************************************************
namespace
{
const int NUM_THREADS = 24;
  const size_t NUM_ELEMENTS = 1.e5;
  const size_t MAX_LOOP_IDX = 1.e4;
  const size_t LOCAL_WORK_SIZE = 256;

  void DotProductHost2(const float* pfData1, const float* pfData2, float* pfResult, int iMin, int iMax)
  {
    int i, k;
    for (i = iMin; i < iMax; i++)
    {
      pfResult[i] = 0.0f;
      for (k = 0; k < 4; k++)
      {
        pfResult[i] += pfData1[4 * i + k] * pfData2[4 * i + k];
      }

      for (int ind = 0; ind < MAX_LOOP_IDX; ind++)
      {
        pfResult[i] += sin(ind * pfData1[ind % (NUM_ELEMENTS)]);
      }

    }
  }

  template <class T>
  void reportConstant(const cl_device_id deviceId, const cl_device_info deviceInfoConstant, std::string msg)
  {
    T info;
    clGetDeviceInfo(deviceId, deviceInfoConstant, sizeof(info), &info, nullptr);
    std::cout << msg << info << std::endl;
  }


  std::vector<int> getLevels(int nMax, int numOfThreads)
  {
    std::vector<int> result;
    result.push_back(0);
    for (int i = 0; i < numOfThreads; ++i)
    {
      double coeff = (i + 1) / double (numOfThreads);
      auto level = (int)round(coeff * nMax);
      result.push_back(level);
    }
    return result;
  
  }
  void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
  {
    auto levels = getLevels(iNumElements, NUM_THREADS);
    std::vector<std::future<void>> futures;
    for (int i = 0; i < levels.size() - 1; i++)
    {
      auto future = std::async(std::launch::async, DotProductHost2, pfData1, pfData2, pfResult, levels[i], levels[i+1]);
      futures.push_back(std::move(future));
    }

    for (auto& f: futures)
    {
      f.get();
    }
    int x = 5; x;

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
    Data(size_t size) : 
      sourceA(size),
      sourceB(size),
      dotProductResults(size)
    {}
    std::vector<cl_float4> sourceA;
    std::vector<cl_float4> sourceB;
    std::vector<cl_float> dotProductResults;
  };

  cl_context createGPUContext(cl_device_id targetDevice)
  {
    return clCreateContext(nullptr, 1, &targetDevice, nullptr, nullptr, nullptr);
  }

  void populateDataInput(Data& data, size_t numElements)
  {
    // Allocate and initialize host arrays
    shrLog("Allocate and Init Host Mem...\n");
    shrFillArray((float*)data.sourceA.data(), int (4 * numElements));
    shrFillArray((float*)data.sourceB.data(), int (4 * numElements));
    shrLog("Allocation done and Init Host Mem...\n");

  }

  bool buildProgram(cl_context gpuContext, cl_device_id targetDevice, cl_program &gpuProgram)
  {
    auto timer = Timer("Build program");

    std::cout << "Creating program" << std::endl;
    size_t programSize = strlen(CL_PROGRAM_DOT_PRODUCT);			// Byte size of kernel code
    gpuProgram = clCreateProgramWithSource(gpuContext, 1, &CL_PROGRAM_DOT_PRODUCT, &programSize, nullptr);
    const char* COMPILATION_FLAGS = "-cl-fast-relaxed-math";
    std::cout << "Building program" << std::endl;
    auto feedback = clBuildProgram(gpuProgram, 0, nullptr, nullptr, nullptr, nullptr);
    if (feedback != CL_SUCCESS)
    {
      oclLogBuildInfo(gpuProgram, targetDevice);
      return false;
    }

    return true;
  }

  DotProductCalculator::Buffers createBuffers(cl_context gpuContext, size_t globalWorkSize)
  {
    auto timer = Timer("Create buffers");

    DotProductCalculator::Buffers buffers;
    buffers.sourceABuffer = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(cl_float4) * globalWorkSize, nullptr, nullptr);
    buffers.sourceBBuffer = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(cl_float4) * globalWorkSize, nullptr, nullptr);
    buffers.dstBuffer = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * globalWorkSize, nullptr, nullptr);

    return buffers;
  }

  cl_kernel createKernel(cl_program gpuProgram, DotProductCalculator::Buffers buffers, size_t numElements)
  {
    auto timer = Timer("Create kernel");

    // Create the kernel
    std::cout << "Creating Kernel (DotProduct)..." << std::endl;
    auto kernel = clCreateKernel(gpuProgram, "DotProduct", nullptr);

    // Set the Argument values
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& buffers.sourceABuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& buffers.sourceBBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& buffers.dstBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)& numElements);

    return kernel;
  }

  cl_command_queue createCommandQueue(
    cl_context gpuContext,
    cl_device_id targetDevice,
    DotProductCalculator::Buffers buffers,
    size_t globalWorkSize,
    const Data& data)
  {
    auto timer = Timer("Create Command Queue and write data to GPU device");
    auto commandQueue = clCreateCommandQueue(gpuContext, targetDevice, 0, nullptr);

    // Asynchronous write of data to GPU device
    clEnqueueWriteBuffer(commandQueue, buffers.sourceABuffer, CL_FALSE, 0, 
      sizeof(cl_float) * globalWorkSize * 4, data.sourceA.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commandQueue, buffers.sourceBBuffer, CL_FALSE, 0, 
      sizeof(cl_float) * globalWorkSize * 4, data.sourceB.data(), 0, nullptr, nullptr);

    return commandQueue;
  }

  void launchKernelAndRun(
    cl_command_queue commandQueue,
    cl_kernel kernel,
    DotProductCalculator::Buffers buffers,
    size_t globalWorkSize,
    size_t localWorkSize,
    std::vector<cl_float>& dotProductResults)
  {
    {
      auto timer = Timer("Run calculation and read back results");
      // Launch kernel
      clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);

      // Read back results and check accumulated errors
      clEnqueueReadBuffer(commandQueue, buffers.dstBuffer, CL_TRUE, 0,
        sizeof(cl_float) * globalWorkSize, dotProductResults.data(), 0, nullptr, nullptr);
    }
  }

  void validateCalculation(const Data& data, size_t numElements)
  {
    // Compute and compare results for golden-host and report errors and pass/fail
    std::vector<cl_float> dotProductResultsValidation(numElements);


    {
      auto timer = Timer("Calculation on CPU");
      DotProductHost((const float*)data.sourceA.data(), (const float*)data.sourceB.data(),
        (float*)dotProductResultsValidation.data(), (int)numElements);
    }

    shrBOOL bMatch = shrComparefet((const float*)dotProductResultsValidation.data(), 
      (const float*)data.dotProductResults.data(), (unsigned int)numElements, 0.0f, 0);
    std::cout << std::boolalpha;
    std::cout << "COMPARING STATUS : " << bMatch << std::endl;
  }
}


void DotProductCalculator::run()
{
  const size_t GLOBAL_WORK_SIZE = shrRoundUp((int)LOCAL_WORK_SIZE, NUM_ELEMENTS);  // rounded up to the nearest multiple of the LocalWorkSize

  auto targetDevice = getTargetDevice();
  reportDeviceInfo(targetDevice);
  reportComputationConstants(NUM_ELEMENTS, GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE);
  Data data(GLOBAL_WORK_SIZE);
  populateDataInput(data, NUM_ELEMENTS);

  gpuContext_ = createGPUContext(targetDevice);

  if (!buildProgram(gpuContext_, targetDevice, gpuProgram_))
    return;

  buffers_ = createBuffers(gpuContext_, GLOBAL_WORK_SIZE);

  kernel_ = createKernel(gpuProgram_, buffers_, NUM_ELEMENTS);

  // --------------------------------------------------------
  // Core sequence... copy input data to GPU, compute, copy results back

  commandQueue_ = createCommandQueue(gpuContext_, targetDevice, buffers_, GLOBAL_WORK_SIZE, data);

  launchKernelAndRun(commandQueue_, kernel_, buffers_,
    GLOBAL_WORK_SIZE, LOCAL_WORK_SIZE, data.dotProductResults);

  validateCalculation(data, NUM_ELEMENTS);

}

DotProductCalculator::~DotProductCalculator()
{
  if (buffers_.sourceABuffer) clReleaseMemObject(buffers_.sourceABuffer);
  if (buffers_.sourceBBuffer) clReleaseMemObject(buffers_.sourceBBuffer);
  if (buffers_.dstBuffer) clReleaseMemObject(buffers_.dstBuffer);
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