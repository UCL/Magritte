/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for read_input.cpp                                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __READ_INPUT_HPP_INCLUDED__
#define __READ_INPUT_HPP_INCLUDED__


#include <string>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* read_txt_input: read the input file                                                           */
/*-----------------------------------------------------------------------------------------------*/

int read_txt_input( std::string inputfile, long ncells, CELL *cell,
                    double *temperature_gas, double *temperature_dust,
                    double *prev_temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



/* read_vtu_input: read the input file                                                           */
/*-----------------------------------------------------------------------------------------------*/

int read_vtu_input( std::string inputfile, long ncells, CELL *cell,
                    double *temperature_gas, double *temperature_dust,
                    double *prev_temperature_gas );

/*-----------------------------------------------------------------------------------------------*/





#endif /* __READ_INPUT_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
