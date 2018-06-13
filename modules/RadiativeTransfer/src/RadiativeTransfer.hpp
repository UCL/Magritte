// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __RADIATIVE_TRANSFER_HPP_INCLUDED__


#include "cells.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"


///  RadiativeTransfer: solves the transfer equation for the radiation field
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] nrays: number of rays that are calculated
///    @param[in] *rays: pointer to the numbers of the rays that are calculated
///    @param[in] lines: data structure containing the line transfer data
///    @param[in] scattering: data structure containing the scattering data
///    @param[in/out] radiation: reference to the  radiation field
////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int RadiativeTransfer (CELLS <Dimension, Nrays>& cells, TEMPERATURE& temperature,
		                   FREQUENCIES& frequencies, const long nrays, const Long1& rays,
											 LINES& lines, SCATTERING& scattering, RADIATION& radiation,
											 Double2& J);


#include "RadiativeTransfer.tpp"


#endif // __RADIATIVE_TRANSFER_HPP_INCLUDED__
