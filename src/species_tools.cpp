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
#include <string.h>

#include <iostream>
#include <string>
using namespace std;

#include "declarations.hpp"
#include "species_tools.hpp"



/* get_canonical_name: get the name of the species as it appears in the species.dat file         */
/*-----------------------------------------------------------------------------------------------*/

string get_canonical_name(string name)
{


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

  cout << "\n WARNING : there is no species with symbol " << canonical_name << "\n";


  /* Set the not found species to be the dummy (zeroth species) */

  spec = 0;

  cout << "\n WARNING : the species " << canonical_name
       << " is set to the \"dummy\" reference with abundance 0.0 \n\n";

  return spec;

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





/* get_charge: get the charge of a species as a multiple of minus the electron charge            */
/*-----------------------------------------------------------------------------------------------*/

// int get_charge(string name)
// {
//
//
//   int charge = 0;                                                       /* charge of the species */
//
//   char list_name[15];
//
//   strcpy(list_name,name.c_str());
//
//   int length = length(name);
//
//   cout << "lengths is " << length << "\n";
//
//
//   for (int letter=0; letter<15; letter++){
//
//     if( strcmp(list_name[letter],"+") ){
//
//       charge++;
//     }
//     else if( strcmp(list_name[letter],"-") ){
//
//       charge--;
//     }
//
//     cout << "letter is " << list_name[letter] << "\n";
//   }
//
//
//   /* get number of + minus the number of - in the expression */
//
//
//   if (charge < 0){
//
//     printf("WARNING: gas is negatively charge even without electrons \n");
//   }
//
//
//   return charge;
//
// }

/*-----------------------------------------------------------------------------------------------*/





/* get_electron_abundance: initialize electron abundance so that the gas is neutral              */
/*-----------------------------------------------------------------------------------------------*/

// int get_electron_abundance(string name)
// {
//
//   for (int spec=0; spec<NSPEC; spec++){
//
//
//   }
//
//
//   return(0);
//
// }

/*-----------------------------------------------------------------------------------------------*/
