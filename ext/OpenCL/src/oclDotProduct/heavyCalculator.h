#pragma once
class HeavyCalculator
{
public:
  void run();
  ~HeavyCalculator();
  struct Buffers
  {
    cl_mem sourceABuffer = 0;
    cl_mem sourceBBuffer = 0;
    cl_mem dstBuffer = 0;
  };
private:
  Buffers buffers_;
  cl_kernel kernel_ = 0;
  cl_program gpuProgram_ = 0;
  cl_command_queue commandQueue_ = 0;
  cl_context gpuContext_ = 0;
};
