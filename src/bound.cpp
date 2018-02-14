// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "reduce.hpp"
#include "initializers.hpp"
#include "HEALPix/chealpix.h"


// bound_cube: put cube boundary around  grid
// ------------------------------------------

long bound_cube (long ncells, CELL *cell_init, CELL *cell_full,
                 long size_x, long size_y, long size_z)
{

  // Add initial cells

# pragma omp parallel                     \
  shared (ncells, cell_init, cell_full)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cell_full[p].x = cell_init[p].x;
    cell_full[p].y = cell_init[p].y;
    cell_full[p].z = cell_init[p].z;

    cell_full[p].density = cell_init[p].density;

    cell_full[p].boundary = false;
  }
  } // end of OpenMP parallel region


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


  double margin_x = 1.0;
  double margin_y = 1.0;
  double margin_z = 1.0; //0.25*length_z;


# if   (DIMENSIONS == 1)

    long n_extra = 2;   // number of boundary cells

    cell_full[NCELLS].x   = x_min - margin_x;
    cell_full[NCELLS].y   = 0.0;
    cell_full[NCELLS].z   = 0.0;

    cell_full[NCELLS].boundary = true;


    cell_full[NCELLS+1].x = x_max + margin_x;
    cell_full[NCELLS+1].y = 0.0;
    cell_full[NCELLS+1].z = 0.0;

    cell_full[NCELLS+1].boundary = true;

# elif (DIMENSIONS == 2)

    long n_extra = 2*(size_x + size_y);   // number of boundary cells

    long index = NCELLS;


    for (int e = 0; e < size_y; e++)
    {
      cell_full[index].x = x_min - margin_x;
      cell_full[index].y = (length_y+2.0*margin_x)/size_y*e + y_min-margin_x;
      cell_full[index].z = 0.0;

      cell_full[index].boundary = true;
      index++;
    }

    for (int e = 0; e < size_y; e++)
    {
      cell_full[index].x = x_max + margin_x;
      cell_full[index].y = -(length_y+2.0*margin_x)/size_y*e + y_max+margin_x;
      cell_full[index].z = 0.0;

      cell_full[index].boundary = true;
      index++;
    }

    for (int e = 0; e < size_x; e++)
    {
      cell_full[index].x = (length_x+2.0*margin_y)/size_x*e + x_min-margin_y;
      cell_full[index].y = y_max + margin_y;
      cell_full[index].z = 0.0;

      cell_full[index].boundary = true;
      cell_full[index].mirror   = true;   // Reflective boundary conditions at upper xz-plane
      index++;
    }

    for (int e = 0; e < size_x; e++)
    {
      cell_full[index].x = -(length_x+2.0*margin_y)/size_x*e + x_max+margin_y;
      cell_full[index].y = y_min - margin_y;
      cell_full[index].z = 0.0;

      cell_full[index].boundary = true;
      index++;
    }


# elif (DIMENSIONS == 3)

    long n_extra = 2*(size_x*size_z + size_y*size_z + size_x*size_y + 1);   // number of boundary cells

    long index = NCELLS;


    for (int e1 = 0; e1 < size_y; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cell_full[index].x = x_min - margin_x;
        cell_full[index].y =  (length_y+2.0*margin_x)/size_y*e1 + y_min-margin_x;
        cell_full[index].z =  (length_z+2.0*margin_x)/size_z*e2 + z_min-margin_x;

        cell_full[index].boundary = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_y; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cell_full[index].x = x_max + margin_x;
        cell_full[index].y = -(length_y+2.0*margin_x)/size_y*e1 + y_max+margin_x;
        cell_full[index].z = -(length_z+2.0*margin_x)/size_z*e2 + z_max+margin_x;;

        cell_full[index].boundary = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cell_full[index].x =  (length_x+2.0*margin_y)/size_x*e1 + x_min-margin_y;
        cell_full[index].y = y_max + margin_y;
        cell_full[index].z = -(length_z+2.0*margin_y)/size_z*e2 + z_max+margin_y;

        cell_full[index].boundary = true;
        cell_full[index].mirror   = true;   // Reflective boundary conditions at upper xz-plane
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cell_full[index].x = -(length_x+2.0*margin_y)/size_x*e1 + x_max+margin_y;
        cell_full[index].y = y_min - margin_y;
        cell_full[index].z =  (length_z+2.0*margin_y)/size_z*e2 + z_min-margin_y;

        cell_full[index].boundary = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_y; e2++)
      {
        cell_full[index].x = (length_x+2.0*margin_z)/size_x*e1 + x_min-margin_z;
        cell_full[index].y = (length_y+2.0*margin_z)/size_y*e2 + y_min-margin_z;
        cell_full[index].z = z_max + margin_z;

        cell_full[index].boundary = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_y; e2++)
      {
        cell_full[index].x = -(length_x+2.0*margin_z)/size_x*e1 + x_max+margin_z;
        cell_full[index].y = -(length_y+2.0*margin_z)/size_y*e2 + y_max+margin_z;
        cell_full[index].z = z_min - margin_z;

        cell_full[index].boundary = true;
        index++;
      }
    }

    cell_full[index].x = x_max + margin_x;
    cell_full[index].y = y_min - margin_y;
    cell_full[index].z = z_max + margin_z;
    index++;

    cell_full[index].x = x_min - margin_x;
    cell_full[index].y = y_max + margin_y;
    cell_full[index].z = z_min - margin_z;

# endif


}




// bound_sphere: put sphere boundary around  grid
// -----------------------------------------------

long bound_sphere (long ncells, CELL *cell_init, CELL *cell_full, long nboundary_cells)
{

  // Add initial cells

# pragma omp parallel                     \
  shared (ncells, cell_init, cell_full)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cell_full[p].x = cell_init[p].x;
    cell_full[p].y = cell_init[p].y;
    cell_full[p].z = cell_init[p].z;
  }
  } // end of OpenMP parallel region


  // Find center

  double x_av = 0.0;
  double y_av = 0.0;
  double z_av = 0.0;


# pragma omp parallel                            \
  shared (ncells, cell_init, x_av, y_av, z_av)   \
  default (none)
  {

# pragma omp for reduction(+ : x_av, y_av, z_av)
  for (long p = 0; p < NCELLS; p++)
  {

    x_av = x_av + cell_init[p].x;
    y_av = y_av + cell_init[p].y;
    z_av = z_av + cell_init[p].z;

  }
  } // end of OpenMP parallel region


  x_av = (double) x_av / NCELLS;
  y_av = (double) y_av / NCELLS;
  z_av = (double) z_av / NCELLS;


  // Find radius

  double radius = 0.0;


# pragma omp parallel                                    \
  shared (ncells, cell_init, radius, x_av, y_av, z_av)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    double radius_new =   (cell_init[p].x-x_av)*(cell_init[p].x-x_av)
                        + (cell_init[p].y-y_av)*(cell_init[p].y-y_av)
                        + (cell_init[p].z-z_av)*(cell_init[p].z-z_av);

    if (radius < radius_new)
    {
      radius = radius_new;
    }

  }
  } // end of OpenMP parallel region


  radius = sqrt(radius);


# if   (DIMENSIONS == 1)

    cell_full[NCELLS].x   = x_av + 1.1*radius;
    cell_full[NCELLS].y   = 0.0;
    cell_full[NCELLS].z   = 0.0;

    cell_full[NCELLS].boundary = true;


    cell_full[NCELLS+1].x = x_av - 1.1*radius;
    cell_full[NCELLS+1].y = 0.0;
    cell_full[NCELLS+1].z = 0.0;

    cell_full[NCELLS+1].boundary = true;

# elif (DIMENSIONS == 2)

    for (long ray = 0; ray < nboundary_cells; ray++)
    {
      double theta = (2.0*PI*ray) / nboundary_cells;

      cell_full[NCELLS+ray].x = x_av + 1.1*radius*cos(theta);
      cell_full[NCELLS+ray].y = y_av + 1.1*radius*sin(theta);
      cell_full[NCELLS+ray].z = 0.0;

      cell_full[NCELLS+ray].boundary = true;
    }

# elif (DIMENSIONS == 3)

    long nsides = (long) sqrt(nboundary_cells/12);

    for (long ipix = 0; ipix < nboundary_cells; ipix++)
    {
      double vector[3];   // unit vector in direction of HEALPix ray

      pix2vec_nest (nsides, ipix, vector);

      cell_full[NCELLS+ipix].x = x_av + 1.1*radius*vector[0];
      cell_full[NCELLS+ipix].y = y_av + 1.1*radius*vector[1];
      cell_full[NCELLS+ipix].z = z_av + 1.1*radius*vector[2];

      cell_full[NCELLS+ipix].boundary = true;
    }

# endif


}
