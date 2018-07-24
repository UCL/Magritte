// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <string>
#include <iostream>
#include <algorithm>

#include "declarations.hpp"
#include "species.hpp"


// Constructor for SPECIES: reads species data file
// ------------------------------------------------

SPECIES::SPECIES (std::string spec_datafile)
{

  // Declare buffers to read data

  int    n;
  char   buffer[BUFFER_SIZE];
  char   sym_buff[15];
  double abn_buff;
  double mass_buff;


  // First species is a dummy for when a species is not found

  sym[0]               = "dummy0";
  initial_abundance[0] = 0.0;


  // Last species is a dummy with abundance 1.0 everywhere

  sym[NSPEC-1]               = "dummy1";
  initial_abundance[NSPEC-1] = 1.0;


  // Open species data file

  FILE *specdata = fopen (spec_datafile.c_str(), "r");

  for (int l = 1; l < NSPEC-1; l++)
  {
    fgets (buffer, BUFFER_SIZE, specdata);
    sscanf (buffer, "%d,%[^,],%lE,%lf %*[^\n] \n", &n, sym_buff, &abn_buff, &mass_buff);

    sym[l]                = sym_buff;
    mass[l]               = mass_buff;
    initial_abundance[l]  = abn_buff;
  }


  // Overwrite electron abundance

  printf ("\n\nNote: Electron abundance is overwritten to make each cell neutral\n\n");

  int electron_nr = SPECIES::get_species_nr ("e-");

  initial_abundance[electron_nr] = 0.0;
  initial_abundance[electron_nr] = SPECIES::get_electron_abundance ();


  fclose (specdata);


  // Get and store species numbers of some inportant species

  nr_e    = SPECIES::get_species_nr ("e-");     // species nr for electrons
  nr_H2   = SPECIES::get_species_nr ("H2");     // species nr for H2
  nr_HD   = SPECIES::get_species_nr ("HD");     // species nr for HD
  nr_C    = SPECIES::get_species_nr ("C");      // species nr for C
  nr_H    = SPECIES::get_species_nr ("H");      // species nr for H
  nr_H2x  = SPECIES::get_species_nr ("H2+");    // species nr for H2+
  nr_HCOx = SPECIES::get_species_nr ("HCO+");   // species nr for HCO+
  nr_H3x  = SPECIES::get_species_nr ("H3+");    // species nr for H3+
  nr_H3Ox = SPECIES::get_species_nr ("H3O+");   // species nr for H3O+
  nr_Hex  = SPECIES::get_species_nr ("He+");    // species nr for He+
  nr_CO   = SPECIES::get_species_nr ("CO");     // species nr for CO


}




// get_canonical_name: get name of species as it appears in species.dat file
// -------------------------------------------------------------------------

std::string SPECIES::get_canonical_name (std::string name)
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

int SPECIES::get_species_nr (std::string name)
{

  // Get the canonical name

  std::string canonical_name = SPECIES::get_canonical_name(name);


  // Chech which species corresponds to canonical name

  for (int spec = 0; spec < NSPEC; spec++)
  {
    if (sym[spec] == canonical_name)
    {
      return spec;
    }

  }


  // If function did not return yet, no match was found

  printf ("\n WARNING : there is no species with symbol %s", canonical_name.c_str());


  // Set not found species to be dummy (zeroth species)

  int spec = 0;

  printf ("\n WARNING : species %s  is set to \"dummy\" reference with abundance 0.0 \n\n",
          canonical_name.c_str());


  return spec;

}




// check_ortho_para: check whether it is ortho or para H2
// ------------------------------------------------------

char SPECIES::check_ortho_para (std::string name)
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

int SPECIES::get_charge (std::string name)
{

  // get number of + minus number of - in spacies name

  return count(name.begin(),name.end(),'+') - count(name.begin(),name.end(),'-');

}




// get_electron_abundance: get electron abundance that would make cell neutral
// ---------------------------------------------------------------------------

double SPECIES::get_electron_abundance ()
{

  double charge_total = 0.0;


  for (int spec = 0; spec < NSPEC; spec++)
  {
    charge_total = charge_total + get_charge(sym[spec])*initial_abundance[spec];
  }


  if (charge_total < 0.0)
  {
    printf ("WARNING: gas is negatively charged even without electrons \n");

    charge_total = 0.0;
  }


  return charge_total;

}
