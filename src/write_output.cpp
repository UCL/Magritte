/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Magritte: writing_output                                                                      */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
#include <sstream>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "write_output.hpp"
#include "write_txt_tools.hpp"
#include "write_vtu_tools.hpp"





/* write_performance_log: write the performance results of the run                               */
/*-----------------------------------------------------------------------------------------------*/

int write_txt_output( double *pop, double *mean_intensity, double *temperature_gas,
                      double *temperature_dust )
{


  std::string tag = "";

  write_abundances(tag);

  write_level_populations(tag, pop);

  write_line_intensities(tag, mean_intensity);

  write_temperature_gas(tag, temperature_gas);

  write_temperature_dust(tag, temperature_dust);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_performance_log: write the performance results of the run                               */
/*-----------------------------------------------------------------------------------------------*/

int write_performance_log( double time_total, double time_level_pop, double time_chemistry,
                           double time_ray_tracing, int n_tb_iterations )
{


  // std::string run_number = RUN_NUMBER;
  // std::string file_name = "tests/performance/performance_" + run_number + ".log";

  std::string file_name = output_directory + "performance.log";

  FILE *file = fopen(file_name.c_str(), "w");


  if (file == NULL)
  {
    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }

  fprintf( file, "NCELLS            %d\n",  NCELLS );
  fprintf( file, "time_total       %lE\n", time_total );
  fprintf( file, "time_ray_tracing %lE\n", time_ray_tracing );
  fprintf( file, "time_level_pop   %lE\n", time_level_pop );
  fprintf( file, "time_chemistry   %lE\n", time_chemistry );
  fprintf( file, "n_tb_iterations  %d\n",  n_tb_iterations );


  fclose(file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
