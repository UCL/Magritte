// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __RADIATIVE_TRANSFER_HPP_INCLUDED__


#include <iostream>


#include "declarations.hpp"
#include "cells.hpp"


template <int Dimension, long Nrays, long Nfreq>
int RadiativeTransfer (CELLS <Dimension, Nrays> *cells, long *freq, long *rays,
											 double *source, double *opacity);


template <int Dimension, long Nrays, long Nfreq>
int set_up_ray (CELLS <Dimension, Nrays> *cells, long o, long r,
							  double *source, double *opacity,
	              long *n, double *S, double *dtau);

#include "RadiativeTransfer.tpp"


#endif // __RADIATIVE_TRANSFER_HPP_INCLUDED__
