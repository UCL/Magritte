/* Frederik De Ceuster - University College London                                               */
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

int write_grid(string tag, GRIDPOINT *gridpoint);

/*-----------------------------------------------------------------------------------------------*/


/* write_healpixvectors: write the unit HEALPix vectors                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_healpixvectors(string tag, double *unit_healpixvector);

/*-----------------------------------------------------------------------------------------------*/



/* write_eval: Write the evaluation points (Z along ray and number of the ray)                   */
/*-----------------------------------------------------------------------------------------------*/

int write_eval(string tag, EVALPOINT *evalpoint);

/*-----------------------------------------------------------------------------------------------*/



/* write_key: write the key to find which grid point corresponds to which evaluation point       */
/*-----------------------------------------------------------------------------------------------*/

int write_key(string tag);

/*-----------------------------------------------------------------------------------------------*/



/* write_raytot: write the total of evaluation points along each ray                             */
/*-----------------------------------------------------------------------------------------------*/

int write_raytot(string tag);

/*-----------------------------------------------------------------------------------------------*/



/* write_cum_raytot: write the cumulative total of evaluation points along each ray              */
/*-----------------------------------------------------------------------------------------------*/

int write_cum_raytot(string tag);

/*-----------------------------------------------------------------------------------------------*/



/* write_abundances: write the abundances at each point                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_abundances(string tag);

/*-----------------------------------------------------------------------------------------------*/



/* write_level_populations: write the level populations at each point for each transition        */
/*-----------------------------------------------------------------------------------------------*/

int write_level_populations(string tag, string *line_datafile, double *pop);

/*-----------------------------------------------------------------------------------------------*/



/* write_line_intensities: write the line intensities for each species, point and transition     */
/*-----------------------------------------------------------------------------------------------*/

int write_line_intensities(string tag, string *line_datafile, double *mean_intensity);

/*-----------------------------------------------------------------------------------------------*/



/* write_temperature_gas: write the gas temperatures at each point                               */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_gas(string tag, double *temperature_gas);

/*-----------------------------------------------------------------------------------------------*/



/* write_temperature_dust: write the dust temperatures at each point                             */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_dust(string tag, double *temperature_dust);

/*-----------------------------------------------------------------------------------------------*/



/* write_UV_field: write the UV field at each point                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_UV_field(string tag, double *UV_field);

/*-----------------------------------------------------------------------------------------------*/



/* write_UV_field: write the UV field at each point                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_AV(string tag, double *AV);

/*-----------------------------------------------------------------------------------------------*/



/* write_rad_surface: write the rad surface at each point                                        */
/*-----------------------------------------------------------------------------------------------*/

int write_rad_surface(string tag, double *rad_surface);

/*-----------------------------------------------------------------------------------------------*/



/* write_reaction_rates: write the rad surface at each point                                        */
/*-----------------------------------------------------------------------------------------------*/

int write_reaction_rates(string tag, REACTION *reaction);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __WRITE_OUTPUT_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
