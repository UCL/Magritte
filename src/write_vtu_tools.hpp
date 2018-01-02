// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRITE_VTU_TOOLS_HPP_INCLUDED__
#define __WRITE_VTU_TOOLS_HPP_INCLUDED__

#include <string>


// write_vtu_output: write all physical variables to vtu input grid
// ----------------------------------------------------------------

int write_vtu_output (std::string inputfile, double *temperature_gas,
                      double *temperature_dust, double *prev_temperature_gas);


#endif // __WRITE_VTU_TOOLS_HPP_INCLUDED__
