#pragma once
class DotProductCalculator
{
public:
  void run();
  ~DotProductCalculator();
private:
  cl_mem srcABuffer_ = 0;
  cl_mem srcBBuffer_ = 0;
  cl_mem dstBuffer_ = 0;
  cl_kernel kernel_ = 0;
  cl_program gpuProgram_ = 0;
  cl_command_queue commandQueue_ = 0;
  cl_context gpuContext_ = 0;
};
