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

CELLS::CELLS (long number_of_cells)
{

  ncells = number_of_cells;


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




// initialize: initialize cells
// ----------------------------

int CELLS::initialize ()
{

# pragma omp parallel   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    x[p] = 0.0;
    y[p] = 0.0;
    z[p] = 0.0;

    n_neighbors[p] = 0;

    for (long r = 0; r < NRAYS; r++)
    {
      neighbor[RINDEX(p,r)] = 0;
      endpoint[RINDEX(p,r)] = 0;

      Z[RINDEX(p,r)]           = 0.0;
      intensity[RINDEX(p,r)]   = 0.0;
      column[RINDEX(p,r)]      = 0.0;
      rad_surface[RINDEX(p,r)] = 0.0;
      AV[RINDEX(p,r)]          = 0.0;
    }

    vx[p] = 0.0;
    vy[p] = 0.0;
    vz[p] = 0.0;

    density[p] = 0.0;

    UV[p] = 0.0;

    for (int s = 0; s < NSPEC; s++)
    {
      abundance[SINDEX(p,s)] = 0.0;
    }

    for (int e = 0; e < NREAC; e++)
    {
      rate[READEX(p,e)] = 0.0;
    }

    for (int l = 0; l < TOT_NLEV; l++)
    {
      pop[LINDEX(p,l)] = 0.0;
    }

    for (int k = 0; k < TOT_NRAD; k++)
    {
      mean_intensity[KINDEX(p,k)] = 0.0;
    }

    temperature_gas[p]      = 10.0;
    temperature_dust[p]     = 10.0;
    temperature_gas_prev[p] =  9.0;

    thermal_ratio[p]      = 1.0;
    thermal_ratio_prev[p] = 1.1;

    id[p] = p;

    removed[p]  = false;
    boundary[p] = false;
    mirror[p]   = false;
  }
  } // end of OpenMP parallel region


  return(0);

}
