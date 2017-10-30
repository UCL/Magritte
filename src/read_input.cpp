/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* read_input: read the input files                                                              */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "read_input.hpp"


/* read_input: read the input file                                                               */
/*-----------------------------------------------------------------------------------------------*/

int read_input(std::string grid_inputfile, GRIDPOINT *gridpoint)
{

  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */


  /* Read input file */

  FILE *input = fopen(grid_inputfile.c_str(), "r");


  /* For all lines in the input file */

  for (long n=0; n<NGRID; n++){

    fgets( buffer, BUFFER_SIZE, input );

    sscanf( buffer, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(gridpoint[n].x), &(gridpoint[n].y), &(gridpoint[n].z),
            &(gridpoint[n].vx), &(gridpoint[n].vy), &(gridpoint[n].vz),
            &(gridpoint[n].density) );
  }


  fclose(input);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
