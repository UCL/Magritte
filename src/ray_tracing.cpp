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
#include "ray_tracing.hpp"
#include "heapsort.hpp"
#include "initializers.hpp"
#include "HEALPix/chealpix.h"


// find_neighbors: find neighboring cells for each cell
// ----------------------------------------------------

int find_neighbors (long ncells, CELLS *cells, RAYS rays)
{

  // For all cells

# pragma omp parallel                      \
  shared (ncells, cells, rays)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    double origin[3];   // cell center of cell p

    origin[0] = cells->x[p];
    origin[1] = cells->y[p];
    origin[2] = cells->z[p];


    // Locate all cell centers w.r.t. origin

#   if (FIXED_NCELLS)

      double ra2[NCELLS];   // squares lengths of local position vectors
      long    rb[NCELLS];   // identifiers of local position vectors

#   else

      double *ra2 = new double[ncells];   // squares lengths of local position vectors
      long   *rb  = new double[ncells];   // identifiers of local position vectors

#   endif


    for (long n = 0; n < NCELLS; n++)
    {
      double rvec[3];   // position vector w.r.t. origin

      rvec[0] = cells->x[n] - origin[0];
      rvec[1] = cells->y[n] - origin[1];
      rvec[2] = cells->z[n] - origin[2];

      ra2[n] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
      rb[n]  = n;
    }


    // Sort cells w.r.t distance from origin

    heapsort (ra2, rb, NCELLS);


    double Z[NRAYS];                 // distance along ray

    initialize_double_array (NRAYS, Z);

    long possible_neighbor[NRAYS];   // cell numbers of neighbors

    initialize_long_array (NRAYS, possible_neighbor);

    bool too_far[NRAYS];             // true when next cell is too far to be a neighbor

    initialize_bool (NRAYS, false, too_far);




    // FIND NEIGHBORS FOR p
    // ____________________


    // Devide the cells over the rays through the origin
    // Start from the second point in rb (first point is cell itself)

    for (long n = 1; n < NCELLS; n++)
    {
      double rvec[3];   // position vector w.r.t. origin

      rvec[0] = cells->x[rb[n]] - origin[0];
      rvec[1] = cells->y[rb[n]] - origin[1];
      rvec[2] = cells->z[rb[n]] - origin[2];


      // Get ray where rvec belongs to (using HEALPix functions)

      long ipix;   // ray index


#     if   (DIMENSIONS == 1)

        if (rvec[0] > 0)
        {
          ipix = 0;
        }
        else
        {
          ipix = 1;
        }

#     elif (DIMENSIONS == 2)

        double theta = acos( rvec[0] / sqrt(ra2[n]) );

        if (rvec[1] >= 0)
        {
          ipix = (long) round(NRAYS * theta / 2.0 / PI);
        }
        else
        {
          ipix = (long) round(NRAYS * (2.0*PI - theta) / 2.0 / PI);
        }

#     elif (DIMENSIONS == 3)

        double theta, phi;   // angles of HEALPix ray

        vec2ang (rvec, &theta, &phi);

        ang2pix_nest (NSIDES, theta, phi, &ipix);

#     endif


      // If there is no neighbor for this ray yet

      if (Z[ipix] == 0.0)
      {
        possible_neighbor[ipix] = rb[n];

        Z[ipix] =   rvec[0]*rays.x[ipix]
                  + rvec[1]*rays.y[ipix]
                  + rvec[2]*rays.z[ipix];

        if (p == 763)
        {
          printf("NEIGHBOUR!\n");
        }

      }

    } // end of n loop over cells (around an origin)


    // Assuming cell boundaries orthogonal to HEALPix ray
    // Check for each possible neighbor if it is too far to be a neighbor

    long index = 0;

    // For all possible neighbors

    for (long pn = 0; pn < NRAYS; pn++)
    {
      if (Z[pn] != 0.0)
      {

        double rvec[3];   // position vector w.r.t. origin

        rvec[0] = cells->x[possible_neighbor[pn]] - origin[0];
        rvec[1] = cells->y[possible_neighbor[pn]] - origin[1];
        rvec[2] = cells->z[possible_neighbor[pn]] - origin[2];


        // For all other possible neighbors

        for (long r = 0; r < NRAYS; r++)
        {
          if ( (Z[r] != 0.0) && (pn != r) )
          {
            double projection =   rvec[0]*rays.x[r]
                                + rvec[1]*rays.y[r]
                                + rvec[2]*rays.z[r];

            if (projection > Z[r]*1.0000001)
            {
              too_far[pn] = true;
            }
          }
        }


        // Enforce that two boundary cells cannot be neighbors

        bool both_boundary = cells->boundary[possible_neighbor[pn]] && cells->boundary[p];


        // If there is a possible neighbor that is not too far

        if ( (Z[pn] != 0.0) && (!too_far[pn]) && !both_boundary )
        {
          cells->neighbor[RINDEX(p,index)] = possible_neighbor[pn];
          index++;
        }
      }
    } // end of pn loop over possible neighbors


    cells->n_neighbors[p] = index;


  } // end of p loop over cells (origins)
  } // end of OpenMP parallel region


# if (!FIXED_NCELLS)

    delete [] ra2;
    delete [] rb;

# endif


  return(0);

}




// next_cell: find number of next cell on ray and its distance along ray
// ---------------------------------------------------------------------

long next_cell (long ncells, CELLS *cells, RAYS rays,
                long origin, long ray, double *Z, long current, double *dZ)
{

  // Pick neighbor on "right side" closest to ray

  double D_min = 1.0E99;

  long next = NCELLS;   // return ncells when there is no next cell


  for (long n = 0; n < cells->n_neighbors[current]; n++)
  {
    long neighbor = cells->neighbor[RINDEX(current,n)];

    double rvec[3];

    rvec[0] = cells->x[neighbor] - cells->x[origin];
    rvec[1] = cells->y[neighbor] - cells->y[origin];
    rvec[2] = cells->z[neighbor] - cells->z[origin];

    double Z_new =   rvec[0]*rays.x[ray]
                   + rvec[1]*rays.y[ray]
                   + rvec[2]*rays.z[ray];

    if (*Z < Z_new)
    {
      double rvec2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

      double D = rvec2 - Z_new*Z_new;

      if (D < D_min)
      {
        D_min = D;
        next  = neighbor;
        *dZ   = Z_new - *Z;   // such that dZ > 0.0
      }
    }

  } // end of n loop over neighbors


  *Z = *Z + *dZ;


  return next;

}




// find_endpoints: find endpoint cells for each cell
// -------------------------------------------------

int find_endpoints (long ncells, CELLS *cells, RAYS rays)
{

  // For all cells

# pragma omp parallel                      \
  shared (ncells, cells, rays)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    for (long r = 0; r < NRAYS; r++)
    {
      double Z  = 0.0;
      double dZ = 0.0;

      long current = p;
      long next    = next_cell (NCELLS, cells, rays, p, r, &Z, current, &dZ);


      while (next != NCELLS)
      {
        current = next;
        next    = next_cell (NCELLS, cells, rays, p, r, &Z, current, &dZ);
      }

      cells->endpoint[RINDEX(p,r)] = current;
      cells->Z[RINDEX(p,r)]        = Z;
    }

  } // end of p loop over cells (origins)
  } // end of OpenMP parallel region


  return(0);

}




// previous_cell: find number of previous cell on ray and its distance along ray
// -----------------------------------------------------------------------------

long previous_cell (long ncells, CELLS *cells, RAYS rays,
                    long origin, long ray, double *Z, long current, double *dZ)
{

  // printf("origin %ld current %ld ray %ld\n", origin, current, ray);
  // printf("Z %lE      dZ %lE\n", *Z, *dZ);

  // Pick neighbor on "right side" closest to ray

  double D_min = 1.0E99;

  long previous = ncells;   // return ncells when there is no previous cell

  double rvec_old[3];

  rvec_old[0] = cells->x[current] - cells->x[origin];
  rvec_old[1] = cells->y[current] - cells->y[origin];
  rvec_old[2] = cells->z[current] - cells->z[origin];

  double rvec_old2 = rvec_old[0]*rvec_old[0] + rvec_old[1]*rvec_old[1] + rvec_old[2]*rvec_old[2];

  for (long n = 0; n < cells->n_neighbors[current]; n++)
  {
    // printf("YESSSS\n");

    long neighbor = cells->neighbor[RINDEX(current,n)];

    // printf("current %ld, nr %ld, n_neighours %ld, neighbour %ld\n", current, n, cell[current].n_neighbors, neighbor);

    double rvec[3];

    rvec[0] = cells->x[neighbor] - cells->x[origin];
    rvec[1] = cells->y[neighbor] - cells->y[origin];
    rvec[2] = cells->z[neighbor] - cells->z[origin];

    // printf("NOOOOOO\n");

    double Z_new =   rvec[0]*rays.x[ray]
                   + rvec[1]*rays.y[ray]
                   + rvec[2]*rays.z[ray];

    double rvec2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

    if ( (*Z > Z_new) && (rvec_old2 > rvec2) )
    {
      double D = rvec2 - Z_new*Z_new;

      if (D < D_min)
      {
        D_min    = D;
        previous = neighbor;
        *dZ      = *Z - Z_new;   // such that dZ > 0.0
      }
    }

  } // end of n loop over neighbors


  *Z = *Z - *dZ;


  return previous;

}




// relative_velocity: get relative velocity of (cell) current w.r.t. (cell) origin along ray
// -----------------------------------------------------------------------------------------

double relative_velocity (long ncells, CELLS *cells, RAYS rays,
                          long origin, long ray, long current)
{

  return   (cells->vx[current] - cells->vx[origin]) * rays.x[ray]
         + (cells->vy[current] - cells->vy[origin]) * rays.y[ray]
         + (cells->vz[current] - cells->vz[origin]) * rays.z[ray];

}
