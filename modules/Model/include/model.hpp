// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MODEL_HPP_INCLUDED__
#define __MODEL_HPP_INCLUDED__


#include <string>
using namespace std;

#include "temperature.hpp"


///  Model: a distributed data structure for Magritte's model data
//////////////////////////////////////////////////////////////////

struct Model
{
  const long dimension;   ///< number of (spacial) dimensions
  const long nrays;       ///< number of rays form each cell
  const long ncells;      ///< number of cells

  Cells       cells;
  Temperature temperature;

  Model (
    const long dimension,
    const long nrays,
    const long ncells);

  int read (const string input_folder);

};

#endif // __MODEL_HPP_INCLUDED__
