// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MODEL_HPP_INCLUDED__
#define __MODEL_HPP_INCLUDED__


#include "types.hpp"
#include "io.hpp"
#include "cells.hpp"
#include "species.hpp"
#include "linedata.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"


///  Model: a distributed data structure for Magritte's model data
//////////////////////////////////////////////////////////////////

struct Model
{

  public:
      // Size (memory)
      long ncells;      ///< number of cells
      long nrays;       ///< number of rays (originating from each cell)
      long nfreqs;      ///< number of frequency bins
      long nspecs;      ///< number of chemical species
      long nlspecs;     ///< number of line producing species

      // Geometry
      Cells       cells;

      // Physical state
      vector <Linedata> linedata;
      Frequencies frequencies;
      Temperature temperature;
      Species     species;


      // Constructor
      Model ();


      // Io
      int read (
          const Io &io);

      int write (
          const Io &io) const;


};


#endif // __MODEL_HPP_INCLUDED__
