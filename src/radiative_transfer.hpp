// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __RADIATIVE_TRANSFER_HPP_INCLUDED__


#include "declarations.hpp"


// radiative_transfer: calculate mean intensity at a cell
// -----------------------------------------------------------

int radiative_transfer (long ncells, CELL *cell, HEALPIXVECTORS healpixvectors, LINES lines,
                        double *Lambda_diagonal, double *mean_intensity_eff,
                        double *Source, double *opacity, long o, int lspec, int kr);


// intensity: calculate intensity along a certain ray through a certain point
// --------------------------------------------------------------------------

int intensities (long ncells, CELL *cell, HEALPIXVECTORS healpixvectors, LINES lines, double *source, double *opacity,
                 double freq, long o, long r, int lspec, int kr,
                 double *u_local, double *v_local, double *L_local);


#endif // __RADIATIVE_TRANSFER_HPP_INCLUDED__
