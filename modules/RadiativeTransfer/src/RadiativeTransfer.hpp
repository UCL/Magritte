// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __RADIATIVE_TRANSFER_HPP_INCLUDED__


#include <iostream>


#include "declarations.hpp"
#include "cells.hpp"
#include "radiation.hpp"
#include "medium.hpp"


template <int Dimension, long Nrays, long Nfreq>
int RadiativeTransfer (CELLS <Dimension, Nrays> *cells, RADIATION *radiation,
											 MEDIUM *medium, long nrays, long *rays, double *J);


template <int Dimension, long Nrays, long Nfreq>
int set_up_ray (CELLS <Dimension, Nrays> *cells, RADIATION *radiation,
		            MEDIUM *medium, long o, long r, long f, double sign,
	              long *n, double *Su, double *Sv, double *dtau);


#include "RadiativeTransfer.tpp"


#endif // __RADIATIVE_TRANSFER_HPP_INCLUDED__
