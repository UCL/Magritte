// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRITE_VTU_TOOLS_HPP_INCLUDED__
#define __WRITE_VTU_TOOLS_HPP_INCLUDED__

#include <string>


// write_vtu_output: write all physical variables to vtu input grid
// ----------------------------------------------------------------

int write_vtu_output (std::string tag, long ncells, CELLS *cells);


#endif // __WRITE_VTU_TOOLS_HPP_INCLUDED__
