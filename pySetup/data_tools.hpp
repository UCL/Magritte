// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SETUP_DATA_TOOLS_HPP_INCLUDED__
#define __SETUP_DATA_TOOLS_HPP_INCLUDED__

#include <string>


// get_NCELLS_vtu: Count number of grid points in .vtu input file
// --------------------------------------------------------------

long get_NCELLS_vtu (std::string inputfile, std::string grid_type);


#endif /* __SETUP_DATA_TOOLS_HPP_INCLUDED__ */
