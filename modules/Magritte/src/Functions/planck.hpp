// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PLANCK_HPP_INCLUDED__
#define __PLANCK_HPP_INCLUDED__


#include "Tools/Parallel/wrap_Grid.hpp"


///  Planck: Planck energy distribution function (in freqiencies)
///    @param[in] temperature_gas: temperature of the gad at this cell
///    @param[in] freq: frequency at which we want evaluate the Planck function
///////////////////////////////////////////////////////////////////////////////

inline vReal planck (
    const double temperature,
    const vReal  freq        );


#include "planck.tpp"


#endif // __PLANCK_HPP_INCLUDED__
