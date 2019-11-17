#pragma once
class DotProductCalculator
{
public:
  void run();
  ~DotProductCalculator();
private:
  cl_mem sourceABuffer_ = 0;
  cl_mem sourceBBuffer_ = 0;
  cl_mem dstBuffer_ = 0;
  cl_kernel kernel_ = 0;
  cl_program gpuProgram_ = 0;
  cl_command_queue commandQueue_ = 0;
  cl_context gpuContext_ = 0;
};
