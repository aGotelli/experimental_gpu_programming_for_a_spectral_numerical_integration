#ifndef TICTOC_H
#define TICTOC_H



#include <chrono>
#include <iostream>
struct tictoc
{
  using Clock = std::chrono::steady_clock;

  inline static void tic() {start = Clock::now();}

  inline static void toc(std::string msg)
  {
    std::cerr << msg << ": " << std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count() <<
                 " ns" << std::endl;
  }
  static Clock::time_point start;
};

tictoc::Clock::time_point tictoc::start;





#endif // TICTOC_H