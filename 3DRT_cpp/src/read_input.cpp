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



/* get_ngrid: Count number of grid points in input file input/ingrid.txt                         */
/*-----------------------------------------------------------------------------------------------*/

long get_ngrid(string inputfile)
{

  long   ngrid=0;                                                       /* number of grid points */


  FILE *input1 = fopen(inputfile.c_str(), "r");

  while ( !feof(input1) && EOF ){

    int ch = fgetc(input1);

    if (ch == '\n'){

      ngrid++;
    }

  }

  fclose(input1);

  return ngrid;

}

/*-----------------------------------------------------------------------------------------------*/



/* read_input: read the input file input/ingrid.txt                                              */
/*-----------------------------------------------------------------------------------------------*/

void read_input(string inputfile, long ngrid, GRIDPOINT *gridpoint )
{

  /* Read input file */

  FILE *input2 = fopen(inputfile.c_str(), "r");

  for (int n=0; n<ngrid; n++){

    if (feof(input2)){

      break;
    }

    fscanf( input2, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(gridpoint[n].x), &(gridpoint[n].y), &(gridpoint[n].z),
            &(gridpoint[n].vx), &(gridpoint[n].vy), &(gridpoint[n].vz) );
  }

  fclose(input2);

}

/*-----------------------------------------------------------------------------------------------*/
