// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINE_PROFILE_HPP_INCLUDED__
#define __LINE_PROFILE_HPP_INCLUDED__

#include "../parameters.hpp"
#include "declarations.hpp"


// line_source: calculate line source function
//--------------------------------------------

int line_source (long ncells, LINE_SPECIES line_species, double *pop, int lspec, double *source);


// line_opacity: calculate line opacity
// ------------------------------------

int line_opacity (long ncells, LINE_SPECIES line_species, double *pop, int lspec, double *opacity);


#if (!CELL_BASED)

  // line_profile: calculate line profile function
  // ---------------------------------------------

  double line_profile (long ncells, CELL *cell, EVALPOINT *evalpoint,
                       double freq, double line_freq, long gridp);

#else

  // line_profile: calculate line profile function
  // ---------------------------------------------

  double cell_line_profile (long ncells, CELL *cell, double velocity,
                            double freq, double line_freq, long gridp);

#endif // if not CELL_BASED


#endif // __LINE_PROFILE_HPP_INCLUDED__
