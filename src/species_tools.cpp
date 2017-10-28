/* Frederik De Ceuster - University College London & KU Leuven                                   */
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
#include <algorithm>

#include "declarations.hpp"
#include "species_tools.hpp"



/* get_canonical_name: get the name of the species as it appears in the species.dat file         */
/*-----------------------------------------------------------------------------------------------*/

std::string get_canonical_name(std::string name)
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

int get_species_nr(std::string name)
{


  int spec;                                                                     /* species index */

  std::string canonical_name = get_canonical_name(name);    /* name as it appears in species.dat */


  /* For all species */

  for (spec=0; spec<NSPEC; spec++){

    if ( species[spec].sym == canonical_name ){


      // cout << "I'm checking " << name << " and I think it is " << spec << "\n";

      return spec;
    }

  }


  /* If the function did not return yet, no match was found */

  printf( "\n WARNING : there is no species with symbol %s", canonical_name.c_str() );


  /* Set the not found species to be the dummy (zeroth species) */

  spec = 0;

  printf( "\n WARNING : the species %s  is set to the \"dummy\" reference with abundance 0.0 \n\n",
          canonical_name.c_str() );


  return spec;

}

/*-----------------------------------------------------------------------------------------------*/





/* check_ortho_para: check whether it is ortho or para H2                                        */
/*-----------------------------------------------------------------------------------------------*/

char check_ortho_para(std::string name)
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

int get_charge(std::string name)
{


  /* using the count standard library function */

  int charge = count(name.begin(),name.end(),'+') - count(name.begin(),name.end(),'-');


  /* get number of + minus the number of - in the expression */


  return charge;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_electron_abundance: initialize electron abundance so that the gas is neutral              */
/*-----------------------------------------------------------------------------------------------*/

double get_electron_abundance(long gridp)
{


  double charge_total = 0.0;


  for (int spec=0; spec<NSPEC; spec++){

    charge_total = charge_total + get_charge(species[spec].sym)*species[spec].abn[gridp];
  }


  if (charge_total < 0.0){

    printf("WARNING: gas is negatively charge even without electrons \n");
  }

  return charge_total;

}

/*-----------------------------------------------------------------------------------------------*/





/* no_better_data: checks whether there data closer to the actual temperature                    */
/*-----------------------------------------------------------------------------------------------*/

bool no_better_data(int reac, REACTION *reaction, double temperature_gas)
{


  bool no_better_data = true;           /* true if there is no better data available in the file */

  int bot_reac = reac - reaction[reac].dup;                   /* first instance of this reaction */
  int top_reac = reac;                                         /* last instance of this reaction */


  while( (reaction[top_reac].dup < reaction[top_reac+1].dup) && (top_reac < NREAC-1) ){

    top_reac = top_reac + 1;
  }


  /* If there are duplicates, look through duplicates for better data */

  if(bot_reac != top_reac){

    for (int rc=bot_reac; rc<=top_reac; rc++){

      double RT_min = reaction[rc].RT_min;
      double RT_max = reaction[rc].RT_max;

      if( (rc != reac) && (RT_min <= temperature_gas) && (temperature_gas <= RT_max) ){

        no_better_data = false;
      }
    }
  }


  return no_better_data;

}

/*-----------------------------------------------------------------------------------------------*/
