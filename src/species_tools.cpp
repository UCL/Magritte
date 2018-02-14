// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <string>
#include <algorithm>

#include "declarations.hpp"
#include "species_tools.hpp"



// get_canonical_name: get name of species as it appears in species.dat file
// -------------------------------------------------------------------------

std::string get_canonical_name (std::string name)
{

  // electrons: e-

  if (name == "e")
  {
    return "e-";
  }


  // di-hydrogen: H2

  if ( (name == "pH2") || (name == "oH2") || (name == "p-H2") || (name == "o-H2") )
  {
    return "H2";
  }


  return name;

}




//  get_species_nr: get number corresponding to given species symbol
// -----------------------------------------------------------------

int get_species_nr (SPECIES *species, std::string name)
{

  std::string canonical_name = get_canonical_name(name);    // name as it appears in species.dat


  // For all species

  for (int spec = 0; spec < NSPEC; spec++)
  {
    if (species[spec].sym == canonical_name)
    {
      return spec;
    }

  }


  // If function did not return yet, no match was found

  printf ("\n WARNING : there is no species with symbol %s", canonical_name.c_str());


  // Set not found species to be dummy (zeroth species)

  int spec = 0;

  printf ("\n WARNING : the species %s  is set to the \"dummy\" reference with abundance 0.0 \n\n",
          canonical_name.c_str());


  return spec;

}




// check_ortho_para: check whether it is ortho or para H2
// ------------------------------------------------------

char check_ortho_para (std::string name)
{

  // ortho-H2

  if ( (name == "oH2") || (name == "o-H2") )
  {
    return 'o';
  }


  // para-H2

  if ( (name == "pH2") || (name == "p-H2") )
  {
    return 'p';
  }


  // If function did not return yet, ortho or para is not relevant

  return 'N';

}




//  get_charge: get charge of a species as a multiple of minus electron charge
// ---------------------------------------------------------------------------

int get_charge (std::string name)
{

  // get number of + minus number of - in spacies name

  return count(name.begin(),name.end(),'+') - count(name.begin(),name.end(),'-');

}




// get_electron_abundance: get electron abundance that would make cell neutral
// ---------------------------------------------------------------------------

double get_electron_abundance (SPECIES *species)
{

  double charge_total = 0.0;


  for (int spec = 0; spec < NSPEC; spec++)
  {
    charge_total = charge_total + get_charge(species[spec].sym)*species[spec].initial_abundance;
  }


  if (charge_total < 0.0)
  {
    printf ("WARNING: gas is negatively charged even without electrons \n");

    charge_total = 0.0;
  }


  return charge_total;

}




// no_better_data: checks whether there data closer to actual temperature
// ----------------------------------------------------------------------

bool no_better_data (int reac, REACTION *reaction, double temperature_gas)
{


  bool no_better_data = true;   // true if there is no better data available in the file

  int bot_reac = reac - reaction[reac].dup;   // first instance of this reaction
  int top_reac = reac;                        // last instance of this reaction


  while ( (reaction[top_reac].dup < reaction[top_reac+1].dup) && (top_reac < NREAC-1) )
  {
    top_reac = top_reac + 1;
  }


  // If there are duplicates, look through duplicates for better data

  if (bot_reac != top_reac)
  {
    for (int rc = bot_reac; rc <= top_reac; rc++)
    {
      double RT_min = reaction[rc].RT_min;
      double RT_max = reaction[rc].RT_max;

      if( (rc != reac) && (RT_min <= temperature_gas) && (temperature_gas <= RT_max) )
      {
        no_better_data = false;
      }
    }
  }


  return no_better_data;

}
