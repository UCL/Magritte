// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __IO_HPP_INCLUDED__
#define __IO_HPP_INCLUDED__


#include "Tools/types.hpp"


///  Io: Magritte's input/output interface
//////////////////////////////////////////

struct Io
{

  const string io_file;


  // Constructor
  Io (
      const string io_file);


  virtual int  read_length   (const string fname,       long    &length) const = 0;

  virtual int  read_number   (const string fname,       long    &number) const = 0;
  virtual int write_number   (const string fname, const long    &number) const = 0;

  virtual int  read_word     (const string fname,       string  &word  ) const = 0;
  virtual int write_word     (const string fname, const string  &word  ) const = 0;

  virtual int  read_list     (const string fname,       Long1   &list  ) const = 0;
  virtual int write_list     (const string fname, const Long1   &list  ) const = 0;

  virtual int  read_list     (const string fname,       Double1 &list  ) const = 0;
  virtual int write_list     (const string fname, const Double1 &list  ) const = 0;

  virtual int  read_list     (const string fname,       String1 &list  ) const = 0;
  virtual int write_list     (const string fname, const String1 &list  ) const = 0;

  virtual int  read_array    (const string fname,       Long2   &array ) const = 0;
  virtual int write_array    (const string fname, const Long2   &array ) const = 0;

  virtual int  read_array    (const string fname,       Double2 &array ) const = 0;
  virtual int write_array    (const string fname, const Double2 &array ) const = 0;

  virtual int  read_3_vector (const string fname,       Double1 &x,
                                                        Double1 &y,
                                                        Double1 &z     ) const = 0;
  virtual int write_3_vector (const string fname, const Double1 &x,
                                                  const Double1 &y,
                                                  const Double1 &z     ) const = 0;

};


#endif // __IO_HPP_INCLUDED__
