/* Frederik De Ceuster - University College London                                               */
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
using namespace std;

#include "declarations.hpp"
#include "read_input.hpp"


/* read_input: read the input file                                                               */
/*-----------------------------------------------------------------------------------------------*/

void read_input(string inputfile, GRIDPOINT *gridpoint)
{

  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */


  /* Read input file */

  FILE *input = fopen(inputfile.c_str(), "r");


  /* For all lines in the input file */

  for (int n=0; n<NGRID; n++){

    fgets( buffer, BUFFER_SIZE, input );

    sscanf( buffer, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(gridpoint[n].x), &(gridpoint[n].y), &(gridpoint[n].z),
            &(gridpoint[n].vx), &(gridpoint[n].vy), &(gridpoint[n].vz),
            &(gridpoint[n].density) );
  }


  fclose(input);

}

/*-----------------------------------------------------------------------------------------------*/
