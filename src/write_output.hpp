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



/* write_grid: write the grid again (for debugging)                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_grid(std::string tag, GRIDPOINT *gridpoint);

/*-----------------------------------------------------------------------------------------------*/


/* write_healpixvectors: write the unit HEALPix vectors                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_healpixvectors(std::string tag, double *unit_healpixvector);

/*-----------------------------------------------------------------------------------------------*/



#ifndef ON_THE_FLY

/* write_eval: Write the evaluation points (Z along ray and number of the ray)                   */
/*-----------------------------------------------------------------------------------------------*/

int write_eval(std::string tag, EVALPOINT *evalpoint);

/*-----------------------------------------------------------------------------------------------*/



/* write_key: write the key to find which grid point corresponds to which evaluation point       */
/*-----------------------------------------------------------------------------------------------*/

int write_key(std::string tag, long *key);

/*-----------------------------------------------------------------------------------------------*/



/* write_raytot: write the total of evaluation points along each ray                             */
/*-----------------------------------------------------------------------------------------------*/

int write_raytot(std::string tag, long *raytot);

/*-----------------------------------------------------------------------------------------------*/



/* write_cum_raytot: write the cumulative total of evaluation points along each ray              */
/*-----------------------------------------------------------------------------------------------*/

int write_cum_raytot(std::string tag, long *cum_raytot);

/*-----------------------------------------------------------------------------------------------*/

#endif



/* write_abundances: write the abundances at each point                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_abundances(std::string tag);

/*-----------------------------------------------------------------------------------------------*/



/* write_level_populations: write the level populations at each point for each transition        */
/*-----------------------------------------------------------------------------------------------*/

int write_level_populations(std::string tag, double *pop);

/*-----------------------------------------------------------------------------------------------*/



/* write_line_intensities: write the line intensities for each species, point and transition     */
/*-----------------------------------------------------------------------------------------------*/

int write_line_intensities(std::string tag, double *mean_intensity);

/*-----------------------------------------------------------------------------------------------*/



/* write_temperature_gas: write the gas temperatures at each point                               */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_gas(std::string tag, double *temperature_gas);

/*-----------------------------------------------------------------------------------------------*/



/* write_temperature_dust: write the dust temperatures at each point                             */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_dust(std::string tag, double *temperature_dust);

/*-----------------------------------------------------------------------------------------------*/



/* write_UV_field: write the UV field at each point                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_UV_field(std::string tag, double *UV_field);

/*-----------------------------------------------------------------------------------------------*/



/* write_UV_field: write the UV field at each point                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_AV(std::string tag, double *AV);

/*-----------------------------------------------------------------------------------------------*/



/* write_rad_surface: write the rad surface at each point                                        */
/*-----------------------------------------------------------------------------------------------*/

int write_rad_surface(std::string tag, double *rad_surface);

/*-----------------------------------------------------------------------------------------------*/



/* write_reaction_rates: write the rad surface at each point                                     */
/*-----------------------------------------------------------------------------------------------*/

int write_reaction_rates(std::string tag, REACTION *reaction);

/*-----------------------------------------------------------------------------------------------*/



/* write_certain_reactions: write rates of certain reactions (as indicated in reaction_rates.cpp)*/
/*-----------------------------------------------------------------------------------------------*/

int write_certain_rates( std::string tag, std::string name, int nr_certain_reac,
                         int *certain_reactions, REACTION *reaction );

/*-----------------------------------------------------------------------------------------------*/



/* write_double_1: write a 1D list of doubles                                                    */
/*-----------------------------------------------------------------------------------------------*/

int write_double_1(std::string name, std::string tag, long length, double *variable);

/*-----------------------------------------------------------------------------------------------*/



/* write_double_2: write a 2D array of doubles                                                   */
/*-----------------------------------------------------------------------------------------------*/

int write_double_2(std::string name, std::string tag, long nrows, long ncols, double *variable);

/*-----------------------------------------------------------------------------------------------*/



/* write_radfield_tools: write the output of the functoins defined in radfield_tools             */
/*-----------------------------------------------------------------------------------------------*/

int write_radfield_tools( std::string tag, double *AV ,double lambda,
                          double *column_H2, double *column_CO );

/*-----------------------------------------------------------------------------------------------*/



/* write_Einstein_coeff: write the Einstein A, B or C coefficients                               */
/*-----------------------------------------------------------------------------------------------*/

int write_Einstein_coeff( std::string tag, double *A_coeff, double *B_coeff, double *C_coeff );

/*-----------------------------------------------------------------------------------------------*/



/* write_R: write the transition matrix R                                                        */
/*-----------------------------------------------------------------------------------------------*/

int write_R( std::string tag, long gridp, double *R );

/*-----------------------------------------------------------------------------------------------*/



/* write_transition_levels: write the levels corresponding to each transition                    */
/*-----------------------------------------------------------------------------------------------*/

int write_transition_levels( std::string tag, int *irad, int *jrad );

/*-----------------------------------------------------------------------------------------------*/



/* write_performance_log: write the performance results of the run                               */
/*-----------------------------------------------------------------------------------------------*/

int write_performance_log( double time_total, double time_level_pop, double time_chemistry,
                           double time_ray_tracing, int niterations );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __WRITE_OUTPUT_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
