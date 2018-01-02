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

int line_source (int *irad, int *jrad, double *A_coeff, double *B_coeff, double *pop, int lspec,
                 double *source);


// line_opacity: calculate line opacity
// ------------------------------------

int line_opacity (int *irad, int *jrad, double *frequency, double *B_coeff, double *pop, int lspec,
                  double *opacity);


#if (!CELL_BASED)


  // line_profile: calculate line profile function
  // ---------------------------------------------

  double line_profile (EVALPOINT *evalpoint, double *temperature_gas, double frequency,
                       double line_frequency, long gridp);


#else


  // line_profile: calculate line profile function
  // ---------------------------------------------

  double cell_line_profile (double velocity, double *temperature_gas, double frequency,
                           double line_frequency, long gridp);


#endif // if not CELL_BASED


#endif // __LINE_PROFILE_HPP_INCLUDED__
