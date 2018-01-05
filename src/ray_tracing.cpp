// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "ray_tracing.hpp"
#include "heapsort.hpp"
#include "initializers.hpp"
#include "HEALPix/chealpix.hpp"


# if (!CELL_BASED)


// find_evalpoints: create evaluation points for each ray from this cell
// ---------------------------------------------------------------------

int find_evalpoints (CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot, long gridp)
{

  // Initialize data structures that store evaluation points

  initialize_long_array(key, NCELLS);
  initialize_long_array(raytot, NRAYS);
  initialize_long_array(cum_raytot, NRAYS);


  // Initialize onray (can still be true from previous call)

  for (long n = 0; n < NCELLS; n++)
  {
    evalpoint[n].onray = false;
  }


  // Place origin at location of cell under consideration

  double origin[3];

  origin[0] = cell[gridp].x;
  origin[1] = cell[gridp].y;
  origin[2] = cell[gridp].z;


  // Locate all cells w.r.t. origin

  double ra2[NCELLS];   // squares of lengths of local position vectors

  long rb[NCELLS];      // cell number corresponding to local position vectors


  for (long n = 0; n < NCELLS; n++)
  {
    double rvec[3];   // Position vector of cell w.r.t. origin

    rvec[0] = cell[n].x - origin[0];
    rvec[1] = cell[n].y - origin[1];
    rvec[2] = cell[n].z - origin[2];

    ra2[n] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
    rb[n]  = n;
  }


  // Sort cells w.r.t distance from origin

  heapsort (ra2, rb, NCELLS);


  double Z[NRAYS];   // distance along ray

  initialize_double_array(Z, NRAYS);


  // FIND EVALUATION POINTS FOR gridp
  // ++++++++++++++++++++++++++++++++


  /* Devide the grid points over the rays through the origin */
  /* Start from the second point in rb (first point is cell itself) */


  for (long n = 1; n < NCELLS; n++)
  {
    double rvec[3];                       /* local position vector of a grid point w.r.t. origin */

    rvec[0] = cell[rb[n]].x - origin[0];
    rvec[1] = cell[rb[n]].y - origin[1];
    rvec[2] = cell[rb[n]].z - origin[2];


    /* Get ray where rvec belongs to (using HEALPix functions) */

    long ipix;                                          /* ray index (as reference to the pixel) */


#   if   (DIMENSIONS == 1)

      if (rvec[0] > 0)
      {
        ipix = 0;
      }
      else
      {
        ipix = 1;
      }

#   elif (DIMENSIONS == 2)

      double theta = acos(rvec[0]/sqrt(ra2[n]));

      if (rvec[1] > 0)
      {
        ipix = (long) round(NRAYS * theta / 2.0 / PI);
      }
      else
      {
        ipix = (long) round(NRAYS * (2.0*PI - theta) / 2.0 / PI);
      }

#   elif (DIMENSIONS == 3)

      double theta, phi;                                            /* angles of the HEALPix ray */

      vec2ang(rvec, &theta, &phi);

      ang2pix_nest(NSIDES, theta, phi, &ipix);


#   endif


    /* Calculate the angle between the cell and its corresponding ray */

    double rvec_dot_uhpv =   rvec[0]*healpixvector[VINDEX(ipix,0)]
	                         + rvec[1]*healpixvector[VINDEX(ipix,1)]
	                         + rvec[2]*healpixvector[VINDEX(ipix,2)];

    double cosine = (rvec_dot_uhpv - Z[ipix])
	                  / sqrt(ra2[n] - 2.0*Z[ipix]*rvec_dot_uhpv + Z[ipix]*Z[ipix]);


    /* Avoid nan angles because of rounding errors */

    if (cosine > 1.0)
    {
      cosine = 1.0;
    }

    double angle = acos( cosine );


    /* If angle < THETA_CRIT, add the new evaluation point */

    if (angle < THETA_CRIT)
    {
      evalpoint[rb[n]].onray = true;
      evalpoint[rb[n]].dZ    = rvec_dot_uhpv - Z[ipix];
      evalpoint[rb[n]].ray   = ipix;
	    evalpoint[rb[n]].Z     = Z[ipix] = rvec_dot_uhpv;
      raytot[ipix]           = raytot[ipix] + 1;
    }

  } /* end of n loop over cells (around an origin) */


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /*   SETUP EVALPOINTS DATA STRUCTURE                                                           */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  cum_raytot[0] = 0;


  for (long r = 1; r < NRAYS; r++)
  {
    cum_raytot[r] = cum_raytot[r-1] + raytot[r-1];
  }


  /* Make a key to find back which evaluation point is where on which ray */

  long nr[NRAYS];                              /* current number of evaluation points on the ray */

  initialize_long_array(nr, NRAYS);


  for (long n = 0; n < NCELLS; n++)
  {
    if (evalpoint[rb[n]].onray == true)
    {
      long ray = evalpoint[rb[n]].ray;

      LOCAL_GP_NR_OF_EVALP(ray, nr[ray]) = rb[n];

      nr[ray] = nr[ray] + 1;
    }
  }


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* get_velocities: get the velocity of the evaluation point with respect to the grid point       */
/*-----------------------------------------------------------------------------------------------*/

int get_velocities (CELL *cell, EVALPOINT *evalpoint,
                    long *key, long *raytot, long *cum_raytot, long gridp, long *first_velo)
{


  /* Since we are in the comoving frame the point itself is at rest */

  evalpoint[gridp].vol = 0.0;


  /* Get the increments in velocity space along each ray/antipodal ray pair */

  for (long r = 0; r < NRAYS/2; r++ )
  {
    long ar = antipod[r];                                         /* index of antipodal ray to r */

    long etot1 = raytot[ar];                   /* total number of evaluation points along ray ar */
    long etot2 = raytot[r];                     /* total number of evaluation points along ray r */

    long ndep = etot1 + etot2;

    // double *velocities;
    // velocities = (double*) malloc( ndep*sizeof(double) );

    // long *evalps;
    // evalps = (long*) malloc( ndep*sizeof(long) );


    if (etot1 > 0)
    {
    for (long e1 = 0; e1 < etot1; e1++)
    {
      long evnr = LOCAL_GP_NR_OF_EVALP(ar,e1);

      evalpoint[evnr].vol
                =   (cell[evnr].vx - cell[gridp].vx) * healpixvector[VINDEX(ar,0)]
                  + (cell[evnr].vy - cell[gridp].vy) * healpixvector[VINDEX(ar,1)]
                  + (cell[evnr].vz - cell[gridp].vz) * healpixvector[VINDEX(ar,2)];

      // velocities[e1] = evalpoint[evnr].vol;

      // evalps[e1]     = evnr;

    } /* end of e loop over evaluation points */
    }


    if (etot2 > 0)
    {
    for (long e2 = 0; e2 < etot2; e2++)
    {
      long evnr = LOCAL_GP_NR_OF_EVALP(r,e2);

      evalpoint[evnr].vol
                =   (cell[evnr].vx - cell[gridp].vx) * healpixvector[VINDEX(r,0)]
                  + (cell[evnr].vy - cell[gridp].vy) * healpixvector[VINDEX(r,1)]
                  + (cell[evnr].vz - cell[gridp].vz) * healpixvector[VINDEX(r,2)];

      // velocities[etot1+e2] = evalpoint[evnr].vol;

      // evalps[etot1+e2]     = evnr;

    } /* end of e loop over evaluation points */
    }


    // /* Sort the velocities by magnitude */
    //
    // heapsort(velocities, evalps, ndep);
    //
    //
    // first_velo[r] = evalps[0];
    //
    //
    // for (long dep=0; dep<ndep-1; dep++){
    //
    //   evalpoint[evalps[dep]].dvc = evalpoint[evalps[dep+1]].vol - evalpoint[evalps[dep]].vol;
    //
    //   evalpoint[evalps[dep]].next_in_velo = evalps[dep+1];
    // }
    //
    //
    // free(velocities);
    // free(evalps);

  } // end of r loop over rays


  return(0);

}


#elif (CELL_BASED)


// find_neighbors: find neighboring cells for each cell
// ----------------------------------------------------

int find_neighbors (long ncells, CELL *cell)
{

  // For all cells

# pragma omp parallel                    \
  shared (ncells, healpixvector, cell)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    double origin[3];   // cell center of cell p

    origin[0] = cell[p].x;
    origin[1] = cell[p].y;
    origin[2] = cell[p].z;


    // Locate all cell centers w.r.t. origin

    double ra2[NCELLS];   // squares lengths of local position vectors

    long rb[NCELLS];      // identifiers of local position vectors


    for (long n = 0; n < ncells; n++)
    {
      double rvec[3];   // position vector w.r.t. origin

      rvec[0] = cell[n].x - origin[0];
      rvec[1] = cell[n].y - origin[1];
      rvec[2] = cell[n].z - origin[2];

      ra2[n] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
      rb[n] = n;
    }


    // Sort cells w.r.t distance from origin

    heapsort(ra2, rb, ncells);


    double Z[NRAYS];                 // distance along ray

    initialize_double_array(Z, NRAYS);

    long possible_neighbor[NRAYS];   // cell numbers of neighbors

    initialize_long_array(possible_neighbor, NRAYS);

    bool too_far[NRAYS];             // true when next cell is too far to be a neighbor

    initialize_bool(false, too_far, NRAYS);




    // FIND NEIGHBORS FOR p
    // ____________________


    // Devide the cells over the rays through the origin
    // Start from the second point in rb (first point is cell itself)

    for (long n = 1; n < ncells; n++)
    {
      double rvec[3];   // position vector w.r.t. origin

      rvec[0] = cell[rb[n]].x - origin[0];
      rvec[1] = cell[rb[n]].y - origin[1];
      rvec[2] = cell[rb[n]].z - origin[2];


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

        double theta = acos(rvec[0]/sqrt(ra2[n]));

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

        vec2ang(rvec, &theta, &phi);

        ang2pix_nest(NSIDES, theta, phi, &ipix);

#     endif


      // If there is no neighbor for this ray yet

      if (Z[ipix] == 0.0)
      {
        possible_neighbor[ipix] = rb[n];

        Z[ipix] =   rvec[0]*healpixvector[VINDEX(ipix,0)]
                  + rvec[1]*healpixvector[VINDEX(ipix,1)]
                  + rvec[2]*healpixvector[VINDEX(ipix,2)];
      }

    } // end of n loop over cells (around an origin)


    // Assuming cell boundaries orthogonal to the HEALPix ray
    // Check for each possible neighbor if it is too far to be a neighbor

    long index = 0;

    // For all possible neighbors

    for (long pn = 0; pn < NRAYS; pn++)
    {
      if (Z[pn] != 0.0)
      {

        double rvec[3];   // position vector w.r.t. origin

        rvec[0] = cell[possible_neighbor[pn]].x - origin[0];
        rvec[1] = cell[possible_neighbor[pn]].y - origin[1];
        rvec[2] = cell[possible_neighbor[pn]].z - origin[2];


        // For all other possible neighbors

        for (long r = 0; r < NRAYS; r++)
        {
          if ( (Z[r] != 0.0) && (pn != r) )
          {
            double projection =   rvec[0]*healpixvector[VINDEX(r,0)]
                                + rvec[1]*healpixvector[VINDEX(r,1)]
                                + rvec[2]*healpixvector[VINDEX(r,2)];

            if (projection > Z[r]*1.0000001)
            {
              too_far[pn] = true;
            }
          }
        }


        // If there is a possible neighbor that is not too far

        if ( (Z[pn] != 0.0) && (!too_far[pn]) )
        {
          cell[p].neighbor[index] = possible_neighbor[pn];
          index++;
        }
      }
    } // end of pn loop over possible neighbors


    cell[p].n_neighbors = index;


  } // end of p loop over cells (origins)
  } // end of OpenMP parallel region


  return(0);

}




// next_cell: find number of next cell on ray and its distance along ray
// ---------------------------------------------------------------------

long next_cell (long ncells, CELL *cell, long origin, long ray, double *Z, long current, double *dZ)
{

  // Pick neighbor on "right side" closest to ray

  double D_min = 1.0E99;

  long next = ncells;   // return ncells when there is no next cell


  for (long n = 0; n < cell[current].n_neighbors; n++)
  {
    long neighbor = cell[current].neighbor[n];

    double rvec[3];

    rvec[0] = cell[neighbor].x - cell[origin].x;
    rvec[1] = cell[neighbor].y - cell[origin].y;
    rvec[2] = cell[neighbor].z - cell[origin].z;

    double Z_new =   rvec[0]*healpixvector[VINDEX(ray,0)]
                   + rvec[1]*healpixvector[VINDEX(ray,1)]
                   + rvec[2]*healpixvector[VINDEX(ray,2)];

    if (*Z < Z_new)
    {
      double rvec2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

      double D = rvec2 - Z_new*Z_new;

      if (D < D_min)
      {
        D_min = D;
        next  = neighbor;
        *dZ   = Z_new - *Z;
      }
    }

  } // end of n loop over neighbors


  *Z = *Z + *dZ;


  return next;

}




// relative_velocity: get relative velocity of (cell) current w.r.t. (cell) origin along ray
// -----------------------------------------------------------------------------------------

double relative_velocity (long ncells, CELL *cell, long origin, long ray, long current)
{

  return   (cell[current].vx - cell[origin].vx) * healpixvector[VINDEX(ray,0)]
         + (cell[current].vy - cell[origin].vy) * healpixvector[VINDEX(ray,1)]
         + (cell[current].vz - cell[origin].vz) * healpixvector[VINDEX(ray,2)];

}


#endif
