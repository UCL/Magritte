// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __INPUT_PYTHON_HPP_INCLUDED__
#define __INPUT_PYTHON_HPP_INCLUDED__


#include "types.hpp"
#include "io.hpp"


///  IoText: io specified by text files
///////////////////////////////////////

struct IoPython : public Io
{

  // Constructor
  IoPython (
      const string io_file);


  // Getters (directly returning)
  long get_number (
      const string file_name) const;

  long get_length (
      const string file_name) const;


  // Readers (implicitly returning) and writers
  int read_list (
      const string file_name,
            Long1 &list      ) const;

  int write_list (
      const string file_name,
      const Long1 &list      ) const;

  int read_array (
      const string file_name,
            Long2 &array     ) const;

  int write_array (
      const string file_name,
      const Long2 &array     ) const;

  int read_list (
      const string file_name,
            Double1 &list    ) const;

  int write_list (
      const string file_name,
      const Double1 &list    ) const;

  int read_array (
      const string file_name,
            Double2 &array   ) const;

  int write_array (
      const string file_name,
      const Double2 &array   ) const;

  int read_3_vector (
      const string   file_name,
            Double1 &x,
            Double1 &y,
            Double1 &z         ) const;

  int write_3_vector (
      const string   file_name,
      const Double1 &x,
      const Double1 &y,
      const Double1 &z         ) const;


};


//#include "../src/io_text.tpp"


#endif // __INPUT_PYTHON_HPP_INCLUDED__
