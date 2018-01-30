// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "reduce.hpp"
#include "initializers.hpp"


// bound_cube: put cube boundary around  grid
// ------------------------------------------

long bound_cube (long ncells, CELL *cell_init, CELL *cell_full, long size)
{


  // Find minimum and maximum coordinate values;

  double x_min = cell_init[0].x;
  double x_max = cell_init[0].x;
  double y_min = cell_init[0].y;
  double y_max = cell_init[0].y;
  double z_min = cell_init[0].z;
  double z_max = cell_init[0].z;


# pragma omp parallel                                                    \
  shared (ncells, cell_init, x_min, x_max, y_min, y_max, z_min, z_max)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    if (x_max < cell_init[p].x)
    {
      x_max = cell_init[p].x;
    }

    if (x_min > cell_init[p].x)
    {
      x_min = cell_init[p].x;
    }

    if (y_max < cell_init[p].y)
    {
      y_max = cell_init[p].y;
    }

    if (y_min > cell_init[p].y)
    {
      y_min = cell_init[p].y;
    }

    if (z_max < cell_init[p].z)
    {
      z_max = cell_init[p].z;
    }

    if (z_min > cell_init[p].z)
    {
      z_min = cell_init[p].z;
    }
  }
  } // end of OpenMP parallel region


  double length_x = x_max - x_min;
  double length_y = y_max - y_min;
  double length_z = z_max - z_min;


# if   (DIMENSIONS == 1)

    long n_extra = 2;                                       // number of boundary cells

    cell_full[NCELLS].x   = x_min - 1.0E-3 * length_x;
    cell_full[NCELLS].y   = 0.0;
    cell_full[NCELLS].z   = 0.0;

    cell_full[NCELLS+1].x = x_max + 1.0E-3 * length_x;
    cell_full[NCELLS+1].y = 0.0;
    cell_full[NCELLS+1].z = 0.0;

# elif (DIMENSIONS == 2)

    long n_extra = 4*(size - 1);                            // number of boundary cells


    for (int e = 0; e < size-1; e++)
    {
      cell_full[NCELLS+e].x = x_min - 1.0E-3 * length_x;
      cell_full[NCELLS+e].y = 1.001*(y_max - y_min)*e + y_min;
      cell_full[NCELLS+e].z = 0.0;
    }

    for (int e = 0; e < size-1; e++)
    {
      cell_full[NCELLS+e].x = x_min - 1.0E-3 * length_x;
      cell_full[NCELLS+e].y = 1.001*(y_max - y_min)/size*(e+1) + y_max;
      cell_full[NCELLS+e].z = 0.0;
    }

    for (int e = 0; e < size-1; e++)
    {
      cell_full[NCELLS+e].x = 1.001*(x_max - x_min)/size*e + x_max;
      cell_full[NCELLS+e].y = y_max + 1.0E-3 * length_y;
      cell_full[NCELLS+e].z = 0.0;
    }

    for (int e = 0; e < size-1; e++)
    {
      cell_full[NCELLS+e].x = 1.001*(x_max - x_min)/size*(e+1) + x_max;
      cell_full[NCELLS+e].y = y_min - 1.0E-3 * length_y;
      cell_full[NCELLS+e].z = 0.0;
    }


# elif (DIMENSIONS == 3)

    long n_extra = 2*size*size + 4*(size - 1)*(size - 2);   // number of boundary cells

# endif


}




// bound_sphere: put sphere boundary around  grid
// -----------------------------------------------

long bound_sphere (long ncells, CELL *cell_init, CELL *cell_full, long nboundary_cells)
{


  // Find center

  double x_av = 0.0;
  double y_av = 0.0;
  double z_av = 0.0;


# pragma omp parallel                            \
  shared (ncells, cell_init, x_av, y_av, z_av)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    x_av = x_av + cell_init[p].x;
    y_av = y_av + cell_init[p].y;
    z_av = z_av + cell_init[p].z;

  }
  } // end of OpenMP parallel region


  x_av = x_av / NCELLS;
  y_av = y_av / NCELLS;
  z_av = z_av / NCELLS;


  // Find radius

  double radius = 0.0;


  # pragma omp parallel                  \
    shared (ncells, cell_init, radius)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {

      double radius_new = cell_init[p].x*cell_init[p].x + cell_init[p].y*cell_init[p].y + cell_init[p].z*cell_init[p].z;

      if (radius < radius_new)
      {
        radius = radius_new;
      }

    }
    } // end of OpenMP parallel region


# if   (DIMENSIONS == 1)

    cell_full[NCELLS].x   = x_av + 1.001*radius;
    cell_full[NCELLS].y   = 0.0;
    cell_full[NCELLS].z   = 0.0;

    cell_full[NCELLS+1].x = x_av - 1.001*radius;
    cell_full[NCELLS+1].y = 0.0;
    cell_full[NCELLS+1].z = 0.0;

# elif (DIMENSIONS == 2)

    for (long ray = 0; ray < nboundary_cells; ray++)
    {
      double theta = (2.0*PI*ray) / nboundary_cells;

      cell_full[NCELLS+ray].x = 1.001*radius*cos(theta);
      cell_full[NCELLS+ray].y = 1.001*radius*sin(theta);
      cell_full[NCELLS+ray].z = 0.0;
    }

# elif (DIMENSIONS == 3)

    long nsides = (long) sqrt(nboundary_cells/12);

    for (long ipix = 0; ipix < nboundary_cells; ipix++)
    {
      double vector[3];   // unit vector in direction of HEALPix ray

      pix2vec_nest (nsides, ipix, vector);

      cell_full[NCELLS+ipix].x = 1.001*radius*vector[0];
      cell_full[NCELLS+ipix].y = 1.001*radius*vector[1];
      cell_full[NCELLS+ipix].z = 1.001*radius*vector[2];
    }

# endif


}
