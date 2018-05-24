#include <iostream>
#include <chrono>


struct TIMER               
{
  std::chrono::duration <double> interval;
  std::chrono::high_resolution_clock::time_point initial;

  void start ()
  {
    initial = std::chrono::high_resolution_clock::now();
  }

  void stop ()
  { 
    interval = std::chrono::high_resolution_clock::now() - initial;
  }

  void print ()
  {
    std::cout << "time  = " << interval.count() << " seconds" << std::endl;
  }

  void print (std::string text)
  {
    std::cout << text << " time  = " << interval.count() << " seconds" << std::endl;
  }
};
