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

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "read_chemdata.hpp"
#include "species_tools.hpp"



// read_species: read species from data file
// -----------------------------------------

int read_species (std::string spec_datafile, SPECIES *species)
{


  char buffer[BUFFER_SIZE];   // buffer for a line of data

  char sym_buff[15];

  double abn_buff;
  double mass_buff;



  // First species is a dummy for when a species is not found

  species[0].sym = "dummy0";

  species[0].initial_abundance = 0.0;


  // Last species is a dummy with abundance 1.0 everywhere

  species[NSPEC-1].sym = "dummy1";

  species[NSPEC-1].initial_abundance = 1.0;


  // Open species data file

  FILE *specdata = fopen (spec_datafile.c_str(), "r");


  for (int l = 1; l < NSPEC-1; l++)
  {
    int n;

    fgets (buffer, BUFFER_SIZE, specdata);
    sscanf (buffer, "%d,%[^,],%lE,%lf %*[^\n] \n", &n, sym_buff, &abn_buff, &mass_buff);

    species[l].sym                = sym_buff;
    species[l].mass               = mass_buff;
    species[l].initial_abundance  = abn_buff;
  }


  // Overwrite electron abindance

  printf("\n\nNote: The electron abundance will be overwritten to make each cell neutral\n\n");

  int electron_nr = get_species_nr (species, "e-");

  species[electron_nr].initial_abundance = 0.0;
  species[electron_nr].initial_abundance = get_electron_abundance (species);


  fclose (specdata);


  // Get and store species numbers of some inportant species

  nr_e    = get_species_nr (species, "e-");     // species nr corresponding to electrons
  nr_H2   = get_species_nr (species, "H2");     // species nr corresponding to H2
  nr_HD   = get_species_nr (species, "HD");     // species nr corresponding to HD
  nr_C    = get_species_nr (species, "C");      // species nr corresponding to C
  nr_H    = get_species_nr (species, "H");      // species nr corresponding to H
  nr_H2x  = get_species_nr (species, "H2+");    // species nr corresponding to H2+
  nr_HCOx = get_species_nr (species, "HCO+");   // species nr corresponding to HCO+
  nr_H3x  = get_species_nr (species, "H3+");    // species nr corresponding to H3+
  nr_H3Ox = get_species_nr (species, "H3O+");   // species nr corresponding to H3O+
  nr_Hex  = get_species_nr (species, "He+");    // species nr corresponding to He+
  nr_CO   = get_species_nr (species, "CO");     // species nr corresponding to CO


  return (0);

}




// read_reactions: read reactoins from (CSV) data file
// ---------------------------------------------------

int read_reactions (std::string reac_datafile, REACTION *reaction)
{

  char *buffer = new char[BUFFER_SIZE];       // buffer for a line of data

  char *buffer_cpy = new char[BUFFER_SIZE];   // buffer for a line of data


  char *alpha_buff  = new char[15];

  char *beta_buff   = new char[15];

  char *gamma_buff  = new char[15];

  char *RT_min_buff = new char[15];

  char *RT_max_buff = new char[15];



  // Open reactions data file

  FILE *reacdata = fopen (reac_datafile.c_str(), "r");


  for (int l = 0; l < NREAC; l++)
  {
    fgets(buffer, BUFFER_SIZE, reacdata);

    buffer_cpy = strdup(buffer);


    // Ignore first column

    strsep(&buffer_cpy, ",");


    // Read first reactant

    reaction[l].R1 = strsep(&buffer_cpy, ",");


    // Read second reactant

    reaction[l].R2 = strsep(&buffer_cpy, ",");


    // Read third reactant

    reaction[l].R3 = strsep(&buffer_cpy, ",");


    // Read first reaction product

    reaction[l].P1 = strsep(&buffer_cpy, ",");


    // Read second reaction product

    reaction[l].P2 = strsep(&buffer_cpy, ",");


    // Read third reaction product

    reaction[l].P3 = strsep(&buffer_cpy, ",");


    // Read fourth reaction product

    reaction[l].P4 = strsep(&buffer_cpy, ",");


    // Read alpha

    alpha_buff = strsep(&buffer_cpy, ",");
    sscanf(alpha_buff, "%lE", &reaction[l].alpha);


    // Read beta

    beta_buff = strsep(&buffer_cpy, ",");
    sscanf(beta_buff, "%lf", &reaction[l].beta);


    // Read gamma

    gamma_buff = strsep(&buffer_cpy, ",");
    sscanf(gamma_buff, "%lf", &reaction[l].gamma);


    // Ignore next column

    strsep(&buffer_cpy, ",");


    RT_min_buff = strsep(&buffer_cpy, ",");
    sscanf(RT_min_buff, "%lf", &reaction[l].RT_min);


    // Read RT_max

    RT_max_buff = strsep(&buffer_cpy, ",");
    sscanf(RT_max_buff, "%lf", &reaction[l].RT_max);


    // Check for duplicates
    // reaction[reac].dup are number of instances of same reaction as reac before reac
    // This is different from 3D-PDR

    reaction[l].dup = 0;

    for (int reac = 0; reac < l; reac++)
    {
      if ( reaction[l].R1 == reaction[reac].R1
           &&  reaction[l].R2 == reaction[reac].R2
           &&  reaction[l].R3 == reaction[reac].R3
           &&  reaction[l].P1 == reaction[reac].P1
           &&  reaction[l].P2 == reaction[reac].P2
           &&  reaction[l].P3 == reaction[reac].P3
           &&  reaction[l].P4 == reaction[reac].P4 )
      {
        reaction[l].dup = reaction[l].dup + 1;
      }
    }

  }

  fclose (reacdata);


  // Free allocated memory

  // delete [] buffer;
  // delete [] buffer_cpy;
  // delete [] alpha_buff;
  // delete [] beta_buff;
  // delete [] gamma_buff;
  // delete [] RT_min_buff;
  // delete [] RT_max_buff;


  return (0);

}
