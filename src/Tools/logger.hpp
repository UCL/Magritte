// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LOGGER_HPP_INCLUDED__
#define __LOGGER_HPP_INCLUDED__


#include <iostream>
#include <fstream>
#include <string>
using std::string;
using std::cout;
using std::endl;

#include "Tools/Parallel/wrap_mpi.hpp"


struct Logger
{

  string file_name;


  Logger ()
  {
    file_name = "magritte_" + std::to_string (MPI_comm_rank()) + ".log";

    std::ofstream file (file_name);
  }

  Logger (const string file_name_local)
  {
    file_name = file_name_local + ".log";

    std::ofstream file (file_name);
  }


  void write (
      const string log_line ) const
  {
    std::ofstream file (file_name, std::ios_base::app);

    file << log_line << endl;
    cout << log_line << endl;
  }

  void write (
      const string text,
      const long   number ) const
  {
    write (text + std::to_string (number));
  }

  void write (
      const string text1,
      const long   number,
      const string text2  ) const
  {
    write (text1 + std::to_string (number) + text2);
  }

  void write (
      const string text,
      const double number ) const
  {
    write (text + std::to_string (number));
  }

  void write (
      const string text1,
      const double number,
      const string text2  ) const
  {
    write (text1 + std::to_string (number) + text2);
  }

  void write_line (void)
  {
    write ("-------------------------------------------------");
  }

};


//inline void write_to_log (
//    const string text    )
//{
//  std::cout << text << std::endl;
//
//
//  std::ofstream file (io_file + file_name + ".txt");
//
//  file << std::scientific << std::setprecision (16);
//
//  file << number;
//
//  file.close();
//
//}
//
//
//inline void write_to_log (
//    const double number  )
//{
//  std::cout << number << std::endl;
//}
//
//
//inline void write_to_log (
//    const string text,
//    const double number  )
//{
//  std::cout << text << number << std::endl;
//}
//
//
//inline void write_to_log (
//    const double number,
//    const string text    )
//{
//  std::cout << number << text << std::endl;
//}
//
//
//inline void write_to_log (
//    const string text1,
//    const double number1,
//    const string text2,
//    const double number2 )
//{
//  std::cout << text1 << number1 << text2 << number2 << std::endl;
//}
//
//
//inline void write_to_log (
//    const string text1,
//    const double number1,
//    const string text2,
//    const double number2,
//    const string text3   )
//{
//  std::cout << text1 << number1 << text2 << number2 << text3 << std::endl;
//}
//
//
//inline void write_to_log (
//    const string text1,
//    const double number1,
//    const string text2,
//    const double number2,
//    const string text3,
//    const double number3 )
//{
//  std::cout << text1 << number1 << text2 << number2 << text3 << number3 << std::endl;
//}


#endif // __LOGGER_HPP_INCLUDED__
