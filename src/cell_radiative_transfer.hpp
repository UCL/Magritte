// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __CELL_RADIATIVE_TRANSFER_HPP_INCLUDED__


#include "../parameters.hpp"
#include "declarations.hpp"

#if (CELL_BASED)


// cell_radiative_transfer: calculate mean intensity at a cell
// -----------------------------------------------------------

int cell_radiative_transfer (long ncells, CELL *cell, double *mean_intensity, double *Lambda_diagonal,
                             double *mean_intensity_eff, double *Source, double *opacity,
                             double *frequency, int *irad, int*jrad, long gridp, int lspec, int kr);


// intensity: calculate intensity along a certain ray through a certain point
// --------------------------------------------------------------------------

int intensities (long ncells, CELL *cell, double *source, double *opacity, double *frequency,
                 double freq, int *irad, int*jrad, long gridp, long r, int lspec, int kr,
                 double *u_local, double *v_local, double *L_local);


#endif


#endif // __CELL_RADIATIVE_TRANSFER_HPP_INCLUDED__
