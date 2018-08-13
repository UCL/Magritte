// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __TEMPERATURE_HPP_INCLUDED__
#define __TEMPERATURE_HPP_INCLUDED__


#include <string>
using namespace std;

#include "types.hpp"


struct TEMPERATURE
{

  const long ncells;

	Double1 gas;        ///< gas temperature
	Double1 dust;       ///< dust temparature
	Double1 gas_prev;   ///< gas temperature in previous iteration


  TEMPERATURE (const long num_of_cells);      ///< Constructor

  int read (const string temperature_file);   ///< read initial temperature field

};


#endif // __TEMPERATURE_HPP_INCLUDED__
