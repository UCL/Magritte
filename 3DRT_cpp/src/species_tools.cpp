/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* species_tools: Some useful functions to find species                                          */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string>
using namespace std;



/* get_species_nr: get the number corresponding to the given species symbol                      */
/*-----------------------------------------------------------------------------------------------*/

int get_species_nr(string name)
{

  int spec;                                                                     /* species index */


  /* For all species */

  for (spec=0; spec<nspec; spec++){

    if ( species[spec].sym == name ){

      return spec;
    }

  }


  /* If the function did not return yet, no match was found */

  printf("ERROR : there is no species with symbol %s \n", name.c_str() );

}

/*-----------------------------------------------------------------------------------------------*/
