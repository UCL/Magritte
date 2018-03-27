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

long bound_cube (long ncells, CELLS *cells_init, CELLS *cells_full,
                 long size_x, long size_y, long size_z)
{

  // Add initial cells

# pragma omp parallel                     \
  shared (ncells, cells_init, cells_full)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cells_full->x[p] = cells_init->x[p];
    cells_full->y[p] = cells_init->y[p];
    cells_full->z[p] = cells_init->z[p];

    cells_full->density[p] = cells_init->density[p];

    cells_full->boundary[p] = false;
  }
  } // end of OpenMP parallel region


  // Find minimum and maximum coordinate values;

  double x_min = cells_init->x[0];
  double x_max = cells_init->x[0];
  double y_min = cells_init->y[0];
  double y_max = cells_init->y[0];
  double z_min = cells_init->z[0];
  double z_max = cells_init->z[0];


# pragma omp parallel                                                     \
  shared (ncells, cells_init, x_min, x_max, y_min, y_max, z_min, z_max)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    if (x_max < cells_init->x[p])
    {
      x_max = cells_init->x[p];
    }

    if (x_min > cells_init->x[p])
    {
      x_min = cells_init->x[p];
    }

    if (y_max < cells_init->y[p])
    {
      y_max = cells_init->y[p];
    }

    if (y_min > cells_init->y[p])
    {
      y_min = cells_init->y[p];
    }

    if (z_max < cells_init->z[p])
    {
      z_max = cells_init->z[p];
    }

    if (z_min > cells_init->z[p])
    {
      z_min = cells_init->z[p];
    }

  }
  } // end of OpenMP parallel region


  double length_x = x_max - x_min;
  double length_y = y_max - y_min;
  double length_z = z_max - z_min;


  double margin_x = 1.0; //E-3 * length_x;
  double margin_y = 1.0; //E-3 * length_y;
  double margin_z = 1.0; //E-3 * length_z; //0.25*length_z;


# if   (DIMENSIONS == 1)

    long n_extra = 2;   // number of boundary cells

    cells_full->x[NCELLS] = x_min - margin_x;
    cells_full->y[NCELLS] = 0.0;
    cells_full->z[NCELLS] = 0.0;

    cells_full->boundary[NCELLS] = true;

    cells_full->x[NCELLS+1] = x_max + margin_x;
    cells_full->y[NCELLS+1] = 0.0;
    cells_full->z[NCELLS+1] = 0.0;

    cells_full->boundary[NCELLS+1] = true;

# elif (DIMENSIONS == 2)

    long n_extra = 2*(size_x + size_y);   // number of boundary cells

    long index = NCELLS;

    for (int e = 0; e < size_y; e++)
    {
      cells_full->x[index] = x_min - margin_x;
      cells_full->y[index] = (length_y+2.0*margin_x)/size_y*e + y_min-margin_x;
      cells_full->z[index] = 0.0;

      cells_full->boundary[index] = true;
      index++;
    }

    for (int e = 0; e < size_y; e++)
    {
      cells_full->x[index] = x_max + margin_x;
      cells_full->y[index] = -(length_y+2.0*margin_x)/size_y*e + y_max+margin_x;
      cells_full->z[index] = 0.0;

      cells_full->boundary[index] = true;
      index++;
    }

    for (int e = 0; e < size_x; e++)
    {
      cells_full->x[index] = (length_x+2.0*margin_y)/size_x*e + x_min-margin_y;
      cells_full->y[index] = y_max + margin_y;
      cells_full->z[index] = 0.0;

      cells_full->boundary[index] = true;
      cells_full->mirror[index]   = true;   // Reflective boundary conditions at upper xz-plane
      index++;
    }

    for (int e = 0; e < size_x; e++)
    {
      cells_full->x[index] = -(length_x+2.0*margin_y)/size_x*e + x_max+margin_y;
      cells_full->y[index] = y_min - margin_y;
      cells_full->z[index] = 0.0;

      cells_full->boundary[index] = true;
      index++;
    }


# elif (DIMENSIONS == 3)

    long n_extra = 2*(size_x*size_z + size_y*size_z + size_x*size_y + 1);   // number of boundary cells

    long index = NCELLS;


    for (int e1 = 0; e1 < size_y; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cells_full->x[index] = x_min - margin_x;
        cells_full->y[index] = (length_y+2.0*margin_x)/size_y*e1 + y_min-margin_x;
        cells_full->z[index] = (length_z+2.0*margin_x)/size_z*e2 + z_min-margin_x;

        cells_full->boundary[index] = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_y; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cells_full->x[index] = x_max + margin_x;
        cells_full->y[index] = -(length_y+2.0*margin_x)/size_y*e1 + y_max+margin_x;
        cells_full->z[index] = -(length_z+2.0*margin_x)/size_z*e2 + z_max+margin_x;;

        cells_full->boundary[index] = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cells_full->x[index] =  (length_x+2.0*margin_y)/size_x*e1 + x_min-margin_y;
        cells_full->y[index] = y_max + margin_y;
        cells_full->z[index] = -(length_z+2.0*margin_y)/size_z*e2 + z_max+margin_y;

        cells_full->boundary[index] = true;
        cells_full->mirror[index]   = true;   // Reflective boundary conditions at upper xz-plane
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_z; e2++)
      {
        cells_full->x[index] = -(length_x+2.0*margin_y)/size_x*e1 + x_max+margin_y;
        cells_full->y[index] = y_min - margin_y;
        cells_full->z[index] =  (length_z+2.0*margin_y)/size_z*e2 + z_min-margin_y;

        cells_full->boundary[index] = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_y; e2++)
      {
        cells_full->x[index] = (length_x+2.0*margin_z)/size_x*e1 + x_min-margin_z;
        cells_full->y[index] = (length_y+2.0*margin_z)/size_y*e2 + y_min-margin_z;
        cells_full->z[index] = z_max + margin_z;

        cells_full->boundary[index] = true;
        index++;
      }
    }

    for (int e1 = 0; e1 < size_x; e1++)
    {
      for (int e2 = 0; e2 < size_y; e2++)
      {
        cells_full->x[index] = -(length_x+2.0*margin_z)/size_x*e1 + x_max+margin_z;
        cells_full->y[index] = -(length_y+2.0*margin_z)/size_y*e2 + y_max+margin_z;
        cells_full->z[index] = z_min - margin_z;

        cells_full->boundary[index] = true;
        index++;
      }
    }

    cells_full->x[index] = x_max + margin_x;
    cells_full->y[index] = y_min - margin_y;
    cells_full->z[index] = z_max + margin_z;
    index++;

    cells_full->x[index] = x_min - margin_x;
    cells_full->y[index] = y_max + margin_y;
    cells_full->z[index] = z_min - margin_z;

# endif


}




// bound_sphere: put sphere boundary around  grid
// -----------------------------------------------

long bound_sphere (long ncells, CELLS *cells_init, CELLS *cells_full, long nboundary_cells)
{

  // Add initial cells

# pragma omp parallel                       \
  shared (ncells, cells_init, cells_full)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cells_full->x[p] = cells_init->x[p];
    cells_full->y[p] = cells_init->y[p];
    cells_full->z[p] = cells_init->z[p];
  }
  } // end of OpenMP parallel region


  // Find center

  double x_av = 0.0;
  double y_av = 0.0;
  double z_av = 0.0;


# pragma omp parallel                             \
  shared (ncells, cells_init, x_av, y_av, z_av)   \
  default (none)
  {

# pragma omp for reduction(+ : x_av, y_av, z_av)
  for (long p = 0; p < NCELLS; p++)
  {

    x_av = x_av + cells_init->x[p];
    y_av = y_av + cells_init->y[p];
    z_av = z_av + cells_init->z[p];

  }
  } // end of OpenMP parallel region


  x_av = (double) x_av / NCELLS;
  y_av = (double) y_av / NCELLS;
  z_av = (double) z_av / NCELLS;


  // Find radius

  double radius = 0.0;


# pragma omp parallel                                     \
  shared (ncells, cells_init, radius, x_av, y_av, z_av)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    double radius_new =   (cells_init->x[p]-x_av)*(cells_init->x[p]-x_av)
                        + (cells_init->y[p]-y_av)*(cells_init->y[p]-y_av)
                        + (cells_init->z[p]-z_av)*(cells_init->z[p]-z_av);

    if (radius < radius_new)
    {
      radius = radius_new;
    }

  }
  } // end of OpenMP parallel region


  radius = sqrt(radius);


# if   (DIMENSIONS == 1)

    cells_full->x[NCELLS] = x_av + 1.1*radius;
    cells_full->y[NCELLS] = 0.0;
    cells_full->z[NCELLS] = 0.0;

    cells_full->boundary[NCELLS] = true;

    cells_full->x[NCELLS+1] = x_av - 1.1*radius;
    cells_full->y[NCELLS+1] = 0.0;
    cells_full->z[NCELLS+1] = 0.0;

    cells_full->boundary[NCELLS+1] = true;

# elif (DIMENSIONS == 2)

    for (long ray = 0; ray < nboundary_cells; ray++)
    {
      double theta = (2.0*PI*ray) / nboundary_cells;

      cells_full->x[NCELLS+ray] = x_av + 1.1*radius*cos(theta);
      cells_full->y[NCELLS+ray] = y_av + 1.1*radius*sin(theta);
      cells_full->z[NCELLS+ray] = 0.0;

      cells_full->boundary[NCELLS+ray] = true;
    }

# elif (DIMENSIONS == 3)

    long nsides = (long) sqrt(nboundary_cells/12);

    for (long ipix = 0; ipix < nboundary_cells; ipix++)
    {
      double vector[3];   // unit vector in direction of HEALPix ray

      pix2vec_nest (nsides, ipix, vector);

      cells_full->x[NCELLS+ipix] = x_av + 1.1*radius*vector[0];
      cells_full->y[NCELLS+ipix] = y_av + 1.1*radius*vector[1];
      cells_full->z[NCELLS+ipix] = z_av + 1.1*radius*vector[2];

      cells_full->boundary[NCELLS+ipix] = true;
    }

# endif


}
