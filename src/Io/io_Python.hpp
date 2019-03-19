// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __INPUT_PYTHON_HPP_INCLUDED__
#define __INPUT_PYTHON_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"


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


      int  read_length   (const string fname,       long    &length) const;

      int  read_number   (const string fname,       long    &number) const;
      int write_number   (const string fname, const long    &number) const;

      int  read_word     (const string fname,       string  &word  ) const;
      int write_word     (const string fname, const string  &word  ) const;

      int  read_list     (const string fname,       Long1   &list  ) const;
      int write_list     (const string fname, const Long1   &list  ) const;

      int  read_list     (const string fname,       Double1 &list  ) const;
      int write_list     (const string fname, const Double1 &list  ) const;

      int  read_list     (const string fname,       String1 &list  ) const;
      int write_list     (const string fname, const String1 &list  ) const;

      int  read_array    (const string fname,       Long2   &array ) const;
      int write_array    (const string fname, const Long2   &array ) const;

      int  read_array    (const string fname,       Double2 &array ) const;
      int write_array    (const string fname, const Double2 &array ) const;

      int  read_3_vector (const string fname,       Double1 &x,
                                                    Double1 &y,
                                                    Double1 &z     ) const;
      int write_3_vector (const string fname, const Double1 &x,
                                              const Double1 &y,
                                              const Double1 &z     ) const;


  private:

      template <class type>
       int read_in_python (
          const string  function,
          const string  file_name,
                type   &data      ) const;

      template <class type>
       int write_in_python (
          const string  function,
          const string  file_name,
          const type   &data      ) const;

};


#endif // __INPUT_PYTHON_HPP_INCLUDED__
