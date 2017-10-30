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



/* read_input: read the input file                                                               */
/*-----------------------------------------------------------------------------------------------*/

int read_input(std::string grid_inputfile, GRIDPOINT *gridpoint);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __READ_INPUT_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
