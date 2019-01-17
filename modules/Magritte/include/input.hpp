// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __INPUT_HPP_INCLUDED__
#define __INPUT_HPP_INCLUDED__


#include "types.hpp"


///  Input: a distributed data structure for Magritte's model data
//////////////////////////////////////////////////////////////////

struct Input
{

  const string input_file;


  // Constructor
  Input (
      const string input_file);


  // Getters (directly returning)
  long get_number (
      const string file_name);

  long get_length (
      const string file_name);


  // Readers (implicitly returning)
  int read_3vector (
      const string   file_name,
            Double1 &x,
            Double1 &y,
            Double1 &z         );

};


#endif // __MODEL_HPP_INCLUDED__
