#pragma once

#include <chrono>
#include <string>

class Timer
{
public:
  Timer(std::string msg);
  ~Timer();
private:
  std::chrono::steady_clock::time_point begin_;
  std::chrono::steady_clock::time_point end_;
  std::string msg_;
};

