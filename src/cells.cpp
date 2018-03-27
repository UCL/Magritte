// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "declarations.hpp"
#include "cells.hpp"


// Constructor for CELLS: reads input file
// ---------------------------------------

CELLS::CELLS (long ncells)
{

# if (!FIXED_NCELLS)


    // Standard input

    x = new double[ncells];
    y = new double[ncells];
    z = new double[ncells];

    vx = new double[ncells];
    vy = new double[ncells];
    vz = new double[ncells];

    density = new double[ncells];


    // First derived UV quantities

    UV          = new double[ncells];
    rad_surface = new double[ncells*NRAYS];
    AV          = new double[ncells*NRAYS];


    // Reduction and interpolation

    id      = new long[ncells];
    removed = new bool[ncells];


    // Boundary conditions

    boundary = new bool[ncells];
    mirror   = new bool[ncells];

    intensity = new double[ncells*NRAYS];
    column    = new double[ncells*NRAYS];


    // Geometry

    endpoint = new long[ncells*NRAYS];
    Z        = new double[ncells*NRAYS];

    neighbor    = new long[ncells*NRAYS];
    n_neighbors = new long[ncells];


    // Chemistry

    abundance = new double[ncells*NSPEC];
    rate      = new double[ncells*NREAC];


    // Lines

    pop            = new double[ncells*TOT_NLEV];
    mean_intensity = new double[ncells*TOT_NRAD];


    // Thermal cell data

    temperature_gas      = new double[ncells];
    temperature_dust     = new double[ncells];
    temperature_gas_prev = new double[ncells];

    thermal_ratio      = new double[ncells];
    thermal_ratio_prev = new double[ncells];


# endif


}




// Destructor for CELLS: frees allocated memory
// --------------------------------------------

CELLS::~CELLS ()
{

# if (!FIXED_NCELLS)


    // Standard input

    delete [] x;
    delete [] y;
    delete [] z;

    delete [] vx;
    delete [] vy;
    delete [] vz;

    delete [] density;


    // First derived UV quantities

    delete [] UV;
    delete [] rad_surface;
    delete [] AV;


    // Reduction and interpolation

    delete [] id;
    delete [] removed;


    // Boundary conditions

    delete [] boundary;
    delete [] mirror;

    delete [] intensity;
    delete [] column;


    // Geometry

    delete [] endpoint;
    delete [] Z;

    delete [] neighbor;
    delete [] n_neighbors;


    // Chemistry

    delete [] abundance;
    delete [] rate;


    // Lines

    delete [] pop;
    delete [] mean_intensity;


    // Thermal cell data

    delete [] temperature_gas;
    delete [] temperature_dust;
    delete [] temperature_gas_prev;

    delete [] thermal_ratio;
    delete [] thermal_ratio_prev;


# endif


}




// read_txt_input: read .txt input file
// ------------------------------------

int CELLS::read_txt_input (std::string inputfile)
{

  char buffer[BUFFER_SIZE];   // buffer for a line of data


  // Read input file

  FILE *input = fopen(inputfile.c_str(), "r");


  // For all lines in input file

  for (long p = 0; p < NCELLS; p++)
  {
    fgets (buffer, BUFFER_SIZE, input);

    sscanf (buffer, "%ld\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf",
            &(id[p]), &(x[p]), &(y[p]), &(z[p]),
            &(vx[p]), &(vy[p]), &(vz[p]),
            &(density[p]));
  }


  fclose(input);


# if (RESTART)


  // Get directory containing restart files

  std::string input_directory = RESTART_DIRECTORY;                  // relative
              input_directory = project_folder + input_directory;   // absolute

  std::string tgas_file_name = input_directory + "temperature_gas.txt";

  if (access (tgas_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *tgas = fopen (tgas_file_name.c_str(), "r");

    for (long p = 0; p < NCELLS; p++)
    {
      fgets (buffer, BUFFER_SIZE, tgas);
      sscanf (buffer, "%lf", &(temperature_gas[p]));
    }

    fclose (tgas);
  }

  else
  {

    // If there is no temperature gas file

    for (long p = 0; p < NCELLS; p++)
    {
      temperature_gas[p] = T_CMB;
    }

  }


  std::string tdust_file_name = input_directory + "temperature_dust.txt";

  if (access (tdust_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *tdust = fopen (tdust_file_name.c_str(), "r");

    for (long p = 0; p < NCELLS; p++)
    {
      fgets (buffer, BUFFER_SIZE, tdust);
      sscanf (buffer, "%lf", &(temperature_dust[p]));
    }

    fclose (tdust);
  }

  else
  {

    // If there is no temperature gas file

    for (long p = 0; p < NCELLS; p++)
    {
      temperature_dust[p] = T_CMB;
    }

  }


  std::string tgas_prev_file_name = input_directory + "temperature_gas_prev.txt";

  if (access (tgas_prev_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *tgas_prev = fopen (tgas_prev_file_name.c_str(), "r");

    for (long p = 0; p < NCELLS; p++)
    {
      fgets (buffer, BUFFER_SIZE, tgas_prev);
      sscanf (buffer, "%lf", &(temperature_gas_prev[p]));
    }

    fclose (tgas_prev);
  }

  else
  {
    // If there is no temperature gas file

    for (long p = 0; p < NCELLS; p++)
    {
      temperature_gas_prev[p] = T_CMB;
    }
  }


  std::string abun_file_name = input_directory + "abundances.txt";

  if (access (abun_file_name.c_str(), F_OK) != -1)
  {

    // If temperature gas file exists

    FILE *abun = fopen (abun_file_name.c_str(), "r");

    for (long p = 0; p < NCELLS; p++)
    {
      for (int s = 0; s < NSPEC; s++)
      {
        fscanf (abun, "%lf", &(abundance[SINDEX(p,s)]));
      }
    }

    fclose (abun);
  }

  else
  {

    // If there is no temperature gas file

    for (long p = 0; p < NCELLS; p++)
    {
      for (int s = 0; s < NSPEC; s++)
      {
        abundance[SINDEX(p,s)] = 0.0;
      }
    }

  }


# endif


  return (0);

}
