// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __RADIATIVE_TRANSFER_HPP_INCLUDED__


#include "cells.hpp"
#include "radiation.hpp"
#include "medium.hpp"


///  RadiativeTransfer: solves the transfer equation for the radiation field
///    @param[in] *cells: pointer to the geometric cell data containing the grid
///    @param[in/out] *radiation: pointer to (previously calculated) radiation field
///    @param[in] *medium: pointer to the opacity and emissivity data of the medium
///    @param[in] nrays: number of rays that are calculated
///    @param[in] *rays: pointer to the numbers of the rays that are calculated
///    @param[out] *J: mean intesity of the radiation field
////////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays, long Nfreq>
int RadiativeTransfer (CELLS <Dimension, Nrays> *cells, RADIATION *radiation,
											 MEDIUM *medium, long nrays, long *rays, double *J);


#include "RadiativeTransfer.tpp"


#endif // __RADIATIVE_TRANSFER_HPP_INCLUDED__
