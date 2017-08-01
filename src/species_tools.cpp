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

#include <iostream>
#include <string>
using namespace std;

#include "declarations.hpp"
#include "species_tools.hpp"



/* get_canonical_name: get the name of the species as it appears in the species.dat file         */
/*-----------------------------------------------------------------------------------------------*/

string get_canonical_name(string name)
{

  string canonical_name;                                    /* name as it appears in species.dat */


  /* electrons: e- */

  if ( name == "e" ){

    return "e-";
  }


  /* di-hydrogen: H2 */

  if ( (name == "pH2")  ||  (name == "oH2")  ||  (name == "p-H2")  ||  (name == "o-H2") ){

    return "H2";
  }


  return name;
}

/*-----------------------------------------------------------------------------------------------*/





/* get_species_nr: get the number corresponding to the given species symbol                      */
/*-----------------------------------------------------------------------------------------------*/

int get_species_nr(string name)
{

  int spec;                                                                     /* species index */

  string canonical_name = get_canonical_name(name);         /* name as it appears in species.dat */


  /* For all species */

  for (spec=0; spec<NSPEC; spec++){

    if ( species[spec].sym == canonical_name ){

      // cout << "I'm checking " << name << " and I think it is " << spec << "\n";

      return spec;
    }

  }


  /* If the function did not return yet, no match was found */

  cout << "ERROR : there is no species with symbol " << name << "\n";

}

/*-----------------------------------------------------------------------------------------------*/





/* check_ortho_para: check whether it is ortho or para H2                                        */
/*-----------------------------------------------------------------------------------------------*/

char check_ortho_para(string name)
{

  /* ortho-H2 */

  if ( (name == "oH2")  ||  (name == "o-H2") ){

    return 'o';
  }


  /* para-H2 */

  if ( (name == "pH2")  ||  (name == "p-H2") ){

    return 'p';
  }


  /* If the function did not return yet, ortho or para is Not relevant */

  // cout << "NOT RELEVANT \n";

  return 'N';
}

/*-----------------------------------------------------------------------------------------------*/
