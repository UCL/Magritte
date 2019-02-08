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

  public:

      const string implementation;


      // Constructor
      IoPython (
          const string implementation,
          const string io_file);


      int read_length (
          const string file_name,
                long  &length    ) const;

      int read_number (
          const string file_name,
                long  &number    ) const;

      int write_number (
          const string file_name,
          const long  &number    ) const;

      int read_word (
          const string  file_name,
                string &word     ) const;

      int write_word (
          const string  file_name,
          const string &word     ) const;

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


  private:

      template <typename type>
      void read_in_python (
          const string  function,
          const string  file_name,
                type   &data      ) const;

      template <typename type>
      void write_in_python (
          const string  function,
          const string  file_name,
          const type   &data      ) const;

};


//#include "../src/io_text.tpp"


#endif // __INPUT_PYTHON_HPP_INCLUDED__
