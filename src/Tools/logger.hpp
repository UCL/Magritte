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
      const string text,
      const size_t number) const
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


#endif // __LOGGER_HPP_INCLUDED__
