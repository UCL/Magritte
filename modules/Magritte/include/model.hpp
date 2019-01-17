// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MODEL_HPP_INCLUDED__
#define __MODEL_HPP_INCLUDED__


#include <string>
using namespace std;

#include "cells.hpp"
#include "species.hpp"
#include "linedata.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"


///  Model: a distributed data structure for Magritte's model data
//////////////////////////////////////////////////////////////////

struct Model
{

//  // Size (memory)
//  const long ncells;      ///< number of cells
//  const long nrays;       ///< number of rays (originating from each cell)
//  const long nfreqs;      ///< number of frequency bins
//  const long nspecs;      ///< number of chemical species
//  const long nlspecs;     ///< number of line producing species

  // Geometry
  Cells       cells;

  // Physical state
  Linedata    linedata;
  Frequencies frequencies;
  Temperature temperature;
  Species     species;

  // Constructor
  Model (
      const string input_folder);


  // Setup and I/O
  int read (
      const string input_folder);

  int write (
      const string output_folder,
      const string tag           ) const;

  int setup ();

};


#endif // __MODEL_HPP_INCLUDED__
