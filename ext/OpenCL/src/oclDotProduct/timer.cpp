#include "timer.h"
#include <iostream>

Timer::Timer(std::string msg)
{
  msg_ = msg;
  begin_ = std::chrono::steady_clock::now();
}

Timer::~Timer()
{
  end_ = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count();
  std::cout << "        !!!Time elapsed for " << msg_ << ": " << ms << "ms" << std::endl;
}
