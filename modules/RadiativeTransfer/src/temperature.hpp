// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __TEMPERATURE_HPP_INCLUDED__
#define __TEMPERATURE_HPP_INCLUDED__


#include <vector>
using namespace std;


struct TEMPERATURE
{
	
  long ncells;

	vector<double> gas;        ///< gas temperature
	
	vector<double> dust;       ///< dust temparature

	vector<double> gas_prev;   ///< gas temperature in previous iteration


  TEMPERATURE (const long num_of_cells);      ///< Constructor

  int read (const string temperature_file);   ///< read initial temperature field

};


#endif // __TEMPERATURE_HPP_INCLUDED__
