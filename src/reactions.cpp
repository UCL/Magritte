// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <string>

#include "declarations.hpp"
#include "reactions.hpp"


// Constructor for REACTIONS
// -------------------------

REACTIONS::REACTIONS (std::string reac_datafile)
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
    fgets (buffer, BUFFER_SIZE, reacdata);

    buffer_cpy = strdup (buffer);


    // Ignore first column

    strsep (&buffer_cpy, ",");


    // Read first reactant

    R1[l] = strsep (&buffer_cpy, ",");


    // Read second reactant

    R2[l] = strsep (&buffer_cpy, ",");


    // Read third reactant

    R3[l] = strsep (&buffer_cpy, ",");


    // Read first reaction product

    P1[l] = strsep (&buffer_cpy, ",");


    // Read second reaction product

    P2[l] = strsep (&buffer_cpy, ",");


    // Read third reaction product

    P3[l] = strsep (&buffer_cpy, ",");


    // Read fourth reaction product

    P4[l] = strsep (&buffer_cpy, ",");


    // Read alpha

    alpha_buff = strsep (&buffer_cpy, ",");
    sscanf (alpha_buff, "%lE", &alpha[l]);


    // Read beta

    beta_buff = strsep (&buffer_cpy, ",");
    sscanf (beta_buff, "%lf", &beta[l]);


    // Read gamma

    gamma_buff = strsep (&buffer_cpy, ",");
    sscanf (gamma_buff, "%lf", &gamma[l]);


    // Ignore next column

    strsep (&buffer_cpy, ",");


    RT_min_buff = strsep (&buffer_cpy, ",");
    sscanf (RT_min_buff, "%lf", &RT_min[l]);


    // Read RT_max

    RT_max_buff = strsep (&buffer_cpy, ",");
    sscanf (RT_max_buff, "%lf", &RT_max[l]);


    // Check for duplicates
    // reaction[reac].dup are number of instances of same reaction as reac before reac
    // This is different from 3D-PDR

    dup[l] = 0;

    for (int reac = 0; reac < l; reac++)
    {
      if (     R1[l] == R1[reac]
           &&  R2[l] == R2[reac]
           &&  R3[l] == R3[reac]
           &&  P1[l] == P1[reac]
           &&  P2[l] == P2[reac]
           &&  P3[l] == P3[reac]
           &&  P4[l] == P4[reac] )
      {
        dup[l] = dup[l] + 1;
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


}




// no_better_data: checks whether there data closer to actual temperature
// ----------------------------------------------------------------------

bool REACTIONS::no_better_data (int reac, double temperature_gas)
{


  bool no_better_data = true;   // true if there is no better data available in the file

  int bot_reac = reac - dup[reac];   // first instance of this reaction
  int top_reac = reac;               // last instance of this reaction


  while ( (dup[top_reac] < dup[top_reac+1]) && (top_reac < NREAC-1) )
  {
    top_reac = top_reac + 1;
  }


  // If there are duplicates, look through duplicates for better data

  if (bot_reac != top_reac)
  {
    for (int rc = bot_reac; rc <= top_reac; rc++)
    {
      if( (rc != reac)
          && (RT_min[rc] <= temperature_gas)
          && (RT_max[rc] >= temperature_gas) )
      {
        no_better_data = false;
      }
    }
  }


  return no_better_data;

}
