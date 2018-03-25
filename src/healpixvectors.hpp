// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __HEALPIXVECTORS_HPP_INCLUDED__
#define __HEALPIXVECTORS_HPP_INCLUDED__

#include <stdio.h>
#include <math.h>

#include "declarations.hpp"
#include "HEALPix/chealpix.h"


struct HEALPIXVECTORS
{

  // Direction cosines of ray

  double x[NRAYS];
  double y[NRAYS];
  double z[NRAYS];


  // Ray number of antipodal ray (for each ray)

  long antipod[NRAYS];


  // Ray mirrored along xz plane (for each ray)

  long mirror_xz[NRAYS];


  // Constructor defines rays based on DIMENSIONS and NRAYS
  // ------------------------------------------------------

  HEALPIXVECTORS ();

};


#endif // __HEALPIXVECTORS_HPP_INCLUDED__
