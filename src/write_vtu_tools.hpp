/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for write_vtu_output.cpp                                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __WRITE_VTU_TOOLS_HPP_INCLUDED__
#define __WRITE_VTU_TOOLS_HPP_INCLUDED__


#include <string>




/* write_vtu_output: write all physical variables to the vtu input grid                          */
/*-----------------------------------------------------------------------------------------------*/

int write_vtu_output( std::string grid_inputfile, double *temperature_gas,
                      double *temperature_dust, double *prev_temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __WRITE_VTU_TOOLS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
