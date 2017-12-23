/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for write_output.cpp                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __WRITE_OUTPUT_HPP_INCLUDED__
#define __WRITE_OUTPUT_HPP_INCLUDED__


#include <string>

#include "declarations.hpp"



/* write_performance_log: write the performance results of the run                               */
/*-----------------------------------------------------------------------------------------------*/

int write_txt_output( double *pop, double *mean_intensity, double *temperature_gas,
                      double *temperature_dust );

/*-----------------------------------------------------------------------------------------------*/



/* write_performance_log: write the performance results of the run                               */
/*-----------------------------------------------------------------------------------------------*/

int write_performance_log( double time_total, double time_level_pop, double time_chemistry,
                           double time_ray_tracing, int niterations );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __WRITE_OUTPUT_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
