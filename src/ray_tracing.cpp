/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* ray_tracing: Create evaluation points along each ray of of each grid point                    */
/*                                                                                               */
/* (based on the evaluation_points routine in 3D-PDR)                                            */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



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



#if !( ON_THE_FLY )

/* ray_tracing: creates the evaluation points for each ray for each grid point                   */
/*-----------------------------------------------------------------------------------------------*/

int ray_tracing( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                 long *key, long *raytot, long *cum_raytot )
{


  long   succes = 0;                                  /* total number of evaluation points found */

  double time_de = 0.0;                                    /* time in dividing evaluation points */
  double time_key = 0.0;                                                 /* time to make the key */
  double time_sort = 0.0;                /* time to sort grid points w.r.t. distance from origin */


  /* Initialize the data structures that will store the evaluation points */

  initialize_long_array(key, NGRID*NGRID);

  initialize_long_array(raytot, NGRID*NRAYS);

  initialize_long_array(cum_raytot, NGRID*NRAYS);


  /* Note: GINDEX, RINDEX and GP_NR_OF_EVALP are defined in definitions.h */


  printf("(ray_tracing): number of grid points     %*d\n", MAX_WIDTH, NGRID);
  printf("(ray_tracing): number of HEALPix sides   %*d\n", MAX_WIDTH, NSIDES);
  printf("(ray_tracing): number of HEALPix rays    %*d\n", MAX_WIDTH, NRAYS);
  printf("(ray_tracing): critical angle THETA_CRIT %*.2lf\n", MAX_WIDTH, THETA_CRIT);
  printf("(ray_tracing): equivalent ray distance   %*.2lE\n", MAX_WIDTH, RAY_SEPARATION2);



  /* For all grid points */

# pragma omp parallel                                                                             \
  shared( healpixvector, gridpoint, evalpoint, key, raytot, cum_raytot, succes,                   \
          time_de, time_key, time_sort )                                                          \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;  /* Note that the brackets are important here */


  for (long gridp=start; gridp<stop; gridp++)
  {

    /* Place the origin at the location of the grid point under consideration */

    double origin[3];                   /* position vector of the grid point under consideration */

    origin[0] = gridpoint[gridp].x;
    origin[1] = gridpoint[gridp].y;
    origin[2] = gridpoint[gridp].z;


    /* Locate all grid points w.r.t. the origin */

    double ra2[NGRID];        /* array with the squares of the lengths of local position vectors */

    long   rb[NGRID];                /* array with the identifiers of the local position vectors */


    for (long n=0; n<NGRID; n++)
    {
      double rvec[3];                     /* local position vector of a grid point w.r.t. origin */

      rvec[0] = gridpoint[n].x - origin[0];
      rvec[1] = gridpoint[n].y - origin[1];
      rvec[2] = gridpoint[n].z - origin[2];

      ra2[n] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];            /* SQUARE length! */
      rb[n]  = n;
    }


    /* Sort the grid points w.r.t their distance from the origin */

    time_sort -= omp_get_wtime();

    heapsort(ra2, rb, NGRID);

    time_sort += omp_get_wtime();


    double Z[NRAYS];                                                       /* distance along ray */

    initialize_double_array(Z, NRAYS);


    /*   FIND EVALUATION POINTS FOR gridp                                                        */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Devide the grid points over the rays through the origin */
    /* Start from the second point in rb (first point is gridpoint itself) */

    time_de -= omp_get_wtime();


    for (long n=1; n<NGRID; n++)
    {
      double rvec[3];                     /* local position vector of a grid point w.r.t. origin */

      rvec[0] = gridpoint[rb[n]].x - origin[0];
      rvec[1] = gridpoint[rb[n]].y - origin[1];
      rvec[2] = gridpoint[rb[n]].z - origin[2];


      /* Get ipix ray where rvec belongs to (using HEALPix functions) */

      double theta, phi;                                            /* angles of the HEALPix ray */

      long   ipix;                                      /* ray index (as reference to the pixel) */

      vec2ang(rvec, &theta, &phi);

      ang2pix_nest(NSIDES, theta, phi, &ipix);


      /* Calculate the angle between the gridpoint and its corresponding ray */

      double rvec_dot_uhpv =   rvec[0]*healpixvector[VINDEX(ipix,0)]
	                           + rvec[1]*healpixvector[VINDEX(ipix,1)]
	                           + rvec[2]*healpixvector[VINDEX(ipix,2)];

      double cosine = (rvec_dot_uhpv - Z[ipix])
		                  / sqrt(ra2[n] - 2.0*Z[ipix]*rvec_dot_uhpv + Z[ipix]*Z[ipix]);


      /* Avoid nan angles because of rounding errors */

      if (cosine>1.0)
      {
        cosine = 1.0;
      }


      double angle = acos( cosine );


      /* If angle < THETA_CRIT, add the new evaluation point */

      if (angle < THETA_CRIT)
      {
        evalpoint[GINDEX(gridp,rb[n])].onray = true;

        evalpoint[GINDEX(gridp,rb[n])].dZ    = rvec_dot_uhpv - Z[ipix];

        evalpoint[GINDEX(gridp,rb[n])].ray   = ipix;

	      evalpoint[GINDEX(gridp,rb[n])].Z     = Z[ipix] = rvec_dot_uhpv;

        evalpoint[GINDEX(gridp,rb[n])].vol
          =   (gridpoint[rb[n]].vx - gridpoint[gridp].vx)*healpixvector[VINDEX(ipix,0)]
            + (gridpoint[rb[n]].vy - gridpoint[gridp].vy)*healpixvector[VINDEX(ipix,1)]
            + (gridpoint[rb[n]].vz - gridpoint[gridp].vz)*healpixvector[VINDEX(ipix,2)];

        succes = succes + 1;

        raytot[RINDEX(gridp,ipix)] = raytot[RINDEX(gridp,ipix)] + 1;


        /* Check whether ipix ray for evaluation point can be considered equivalent */

        double distance_to_ray2 = ra2[n] - rvec_dot_uhpv * rvec_dot_uhpv;

        if (distance_to_ray2 < RAY_SEPARATION2)
        {
          evalpoint[GINDEX(gridp,rb[n])].eqp = gridp;
        }

        else
        {
          evalpoint[GINDEX(gridp,rb[n])].eqp = rb[n];
        }

      } /* end of if angle < THETA_CRIT */

    } /* end of n loop over gridpoints (around an origin) */


    time_de += omp_get_wtime();


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   SETUP EVALPOINTS DATA STRUCTURE                                                         */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    cum_raytot[RINDEX(gridp,0)] = 0;


    for (long r=1; r<NRAYS; r++)
    {
      cum_raytot[RINDEX(gridp,r)] = cum_raytot[RINDEX(gridp,r-1)] + raytot[RINDEX(gridp,r-1)];
    }


    /* Make a key to find back which evaluation point is where on which ray */

    long nr[NRAYS];                        /* current number of evaluation points along each ray */

    initialize_long_array(nr, NRAYS);

    time_key -= omp_get_wtime();


    for (long n=0; n<NGRID; n++)
    {
      if (evalpoint[GINDEX(gridp,rb[n])].onray == true)
      {
        long ray = evalpoint[GINDEX(gridp,rb[n])].ray;

        GP_NR_OF_EVALP(gridp, ray, nr[ray]) = rb[n];

        nr[ray] = nr[ray] + 1;
      }
    }

    time_key += omp_get_wtime();


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  } /* end of gridp loop over grid points (origins) */
  } /* end of OpenMP parallel region */


  printf( "(ray_tracing): time in dividing evaluation points %lf sec\n", time_de );
  printf( "(ray_tracing): time in making the key             %lf sec\n", time_key );
  printf( "(ray_tracing): time in heapsort routine           %lf sec\n", time_sort );

  printf( "(ray_tracing): succes rate %.2lf" \
          "(# eval. points)/ ((# grid points)^2 - (# grid points)) \n",
          (double) succes/(NGRID*NGRID-NGRID) );


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





#else

/* get_local_evalpoint: creates the evaluation points for each ray for this grid point           */
/*-----------------------------------------------------------------------------------------------*/

int get_local_evalpoint( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                         long *key, long *raytot, long *cum_raytot, long gridp )
{


  /* Initialize the data structures that will store the evaluation points */

  initialize_long_array(key, NGRID);
  initialize_long_array(raytot, NRAYS);
  initialize_long_array(cum_raytot, NRAYS);


  /* Initialize on ray, might still be true from previous call to get_local_evalpoint */

  for (long n=0; n<NGRID; n++)
  {
    evalpoint[n].onray = false;
  }


  /* Place the origin at the location of the grid point under consideration */

  double origin[3];                     /* position vector of the grid point under consideration */

  origin[0] = gridpoint[gridp].x;
  origin[1] = gridpoint[gridp].y;
  origin[2] = gridpoint[gridp].z;


  /* Locate all grid points w.r.t. the origin */

  double ra2[NGRID];          /* array with the squares of the lengths of local position vectors */

  long rb[NGRID];                    /* array with the identifiers of the local position vectors */


  for (long n=0; n<NGRID; n++)
  {
    double rvec[3];                       /* local position vector of a grid point w.r.t. origin */

    rvec[0] = gridpoint[n].x - origin[0];
    rvec[1] = gridpoint[n].y - origin[1];
    rvec[2] = gridpoint[n].z - origin[2];

    ra2[n] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];              /* SQUARE length! */
    rb[n]  = n;
  }


  /* Sort the grid points w.r.t their distance from the origin */

  heapsort(ra2, rb, NGRID);


  double Z[NRAYS];                                                         /* distance along ray */

  initialize_double_array(Z, NRAYS);


  /*   FIND EVALUATION POINTS FOR gridp                                                          */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  /* Devide the grid points over the rays through the origin */
  /* Start from the second point in rb (first point is gridpoint itself) */


  for (long n=1; n<NGRID; n++)
  {
    double rvec[3];                       /* local position vector of a grid point w.r.t. origin */

    rvec[0] = gridpoint[rb[n]].x - origin[0];
    rvec[1] = gridpoint[rb[n]].y - origin[1];
    rvec[2] = gridpoint[rb[n]].z - origin[2];


    /* Get ipix ray where rvec belongs to (using HEALPix functions) */

    double theta, phi;                                              /* angles of the HEALPix ray */

    long   ipix;                                        /* ray index (as reference to the pixel) */

    vec2ang(rvec, &theta, &phi);

    ang2pix_nest(NSIDES, theta, phi, &ipix);


    /* Calculate the angle between the gridpoint and its corresponding ray */

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


      /* Check whether ipix ray for evaluation point can be considered equivalent */

      double distance_to_ray2 = ra2[n] - rvec_dot_uhpv*rvec_dot_uhpv;

      if (distance_to_ray2 < RAY_SEPARATION2)
      {
        evalpoint[rb[n]].eqp = gridp;
      }

      else
      {
        evalpoint[rb[n]].eqp = rb[n];
      }

    } /* end of if angle < THETA_CRIT */

  } /* end of n loop over gridpoints (around an origin) */


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /*   SETUP EVALPOINTS DATA STRUCTURE                                                           */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  cum_raytot[0] = 0;


  for (long r=1; r<NRAYS; r++)
  {
    cum_raytot[r] = cum_raytot[r-1] + raytot[r-1];
  }


  /* Make a key to find back which evaluation point is where on which ray */

  long nr[NRAYS];                              /* current number of evaluation points on the ray */

  initialize_long_array(nr, NRAYS);


  for (long n=0; n<NGRID; n++)
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

int get_velocities( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                    long *key, long *raytot, long *cum_raytot, long gridp, long *first_velo )
{


  /* Since we are in the comoving frame the point itself is at rest */

  evalpoint[gridp].vol = 0.0;


  /* Get the increments in velocity space along each ray/antipodal ray pair */

  for (long r=0; r<NRAYS/2; r++ )
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
    for (long e1=0; e1<etot1; e1++)
    {
      long evnr = LOCAL_GP_NR_OF_EVALP(ar,e1);

      evalpoint[evnr].vol
                =   (gridpoint[evnr].vx - gridpoint[gridp].vx) * healpixvector[VINDEX(ar,0)]
                  + (gridpoint[evnr].vy - gridpoint[gridp].vy) * healpixvector[VINDEX(ar,1)]
                  + (gridpoint[evnr].vz - gridpoint[gridp].vz) * healpixvector[VINDEX(ar,2)];

      // velocities[e1] = evalpoint[evnr].vol;

      // evalps[e1]     = evnr;

    } /* end of e loop over evaluation points */
    }


    if (etot2 > 0)
    {
    for (long e2=0; e2<etot2; e2++)
    {
      long evnr = LOCAL_GP_NR_OF_EVALP(r,e2);

      evalpoint[evnr].vol
                =   (gridpoint[evnr].vx - gridpoint[gridp].vx) * healpixvector[VINDEX(r,0)]
                  + (gridpoint[evnr].vy - gridpoint[gridp].vy) * healpixvector[VINDEX(r,1)]
                  + (gridpoint[evnr].vz - gridpoint[gridp].vz) * healpixvector[VINDEX(r,2)];

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

  } /* end of r loop over rays */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/

#endif



/* find_neighbors: creates the evaluation points for each ray for each cell                      */
/*-----------------------------------------------------------------------------------------------*/

int find_neighbors( long ncells, CELL *cell )
{


  /* For all cells */

# pragma omp parallel                                                                             \
  shared( ncells, healpixvector, cell )                                                           \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;     /* Note that brackets are important here */


  for (long p=start; p<stop; p++)
  {

    /* Place the origin at the location of the cell under consideration */

    double origin[3];                   /* position vector of the cell under consideration */

    origin[0] = cell[p].x;
    origin[1] = cell[p].y;
    origin[2] = cell[p].z;


    /* Locate all cell centers w.r.t. the origin */

    double *ra2 = new double[ncells];               /* squares lengths of local position vectors */

    long    *rb = new long[ncells];                     /* identifiers of local position vectors */


    for (long n=0; n<ncells; n++)
    {
      double rvec[3];                    /* local position vector of a cell center w.r.t. origin */

      rvec[0] = cell[n].x - origin[0];
      rvec[1] = cell[n].y - origin[1];
      rvec[2] = cell[n].z - origin[2];

      ra2[n] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];            /* SQUARE length! */
      rb[n]  = n;
    }


    /* Sort the cells w.r.t their distance from the origin */

    heapsort(ra2, rb, ncells);


    double Z[NRAYS];                                                       /* distance along ray */

    initialize_double_array(Z, NRAYS);

    long possible_neighbor[NRAYS];                                  /* cell numbers of neighbors */

    initialize_long_array(possible_neighbor, NRAYS);

    bool too_far[NRAYS];

    initialize_bool(false, NRAYS, too_far);


    /*   FIND NEIGHBORS FOR p                                                                    */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Devide the cells over the rays through the origin */
    /* Start from the second point in rb (first point is cell itself) */

    for (long n=1; n<ncells; n++)
    {
      double rvec[3];                    /* local position vector of a cell center w.r.t. origin */

      rvec[0] = cell[rb[n]].x - origin[0];
      rvec[1] = cell[rb[n]].y - origin[1];
      rvec[2] = cell[rb[n]].z - origin[2];


      /* Get ray where rvec belongs to (using HEALPix functions) */

      double theta, phi;                                            /* angles of the HEALPix ray */

      long ipix;                                        /* ray index (as reference to the pixel) */

      long nsides = (long) sqrt(NRAYS/12);


      vec2ang(rvec, &theta, &phi);

      ang2pix_nest(nsides, theta, phi, &ipix);


      /* If there is no neighbor for this ray yet */

      if (Z[ipix] == 0.0)
      {
        possible_neighbor[ipix] = rb[n];

        Z[ipix] =   rvec[0]*healpixvector[VINDEX(ipix,0)]
                  + rvec[1]*healpixvector[VINDEX(ipix,1)]
                  + rvec[2]*healpixvector[VINDEX(ipix,2)];
      }

    } /* end of n loop over cells (around an origin) */


    /* Assuming cell boundaries orthogonal to the HEALPix ray */
    /* Check along which ray the next point is too far to be a neighbor */

    long index = 0;


    for (long ray=0; ray<NRAYS; ray++)
    {
      double rvec[3];                    /* local position vector of a cell center w.r.t. origin */

      rvec[0] = cell[possible_neighbor[ray]].x - origin[0];
      rvec[1] = cell[possible_neighbor[ray]].y - origin[1];
      rvec[2] = cell[possible_neighbor[ray]].z - origin[2];

      for (long r=0; r<NRAYS; r++)
      {
        double projection =   rvec[0]*healpixvector[VINDEX(r,0)]
                            + rvec[1]*healpixvector[VINDEX(r,1)]
                            + rvec[2]*healpixvector[VINDEX(r,2)];

        if ( (ray != r) && (Z[r] != 0.0) && (projection >= Z[r]) )
        {
          too_far[ray] = true;
        }
      }

      if ( !too_far[ray] )
      {
        cell[p].neighbor[index] = possible_neighbor[ray];
        index++;
      }

    } /* end of ray loop over rays */


    cell[p].n_neighbors = index;

    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  } /* end of p loop over cells (origins) */
  } /* end of OpenMP parallel region */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/



/* next_cell_on_ray: find the number of the next cell on the ray and its distance along the ray  */
/*-----------------------------------------------------------------------------------------------*/

int next_cell_on_ray( long ncells, CELL *cell, long current, long origin, long ray, double Z,
                      long *next, double *dZ )
{

  /* Pick the neighbor on the "right side" closesd to the ray */

  double D_min = 1.0E99;

  *next = ncells;

  for (long n=0; n<cell[current].n_neighbors; n++)
  {
    long neighbor = cell[current].neighbor[n];

    double rvec[3];

    rvec[0] = cell[neighbor].x - cell[origin].x;
    rvec[1] = cell[neighbor].y - cell[origin].y;
    rvec[2] = cell[neighbor].z - cell[origin].z;

    double new_Z =   rvec[0]*healpixvector[VINDEX(ray,0)]
                   + rvec[1]*healpixvector[VINDEX(ray,1)]
                   + rvec[2]*healpixvector[VINDEX(ray,2)];

    if (Z < new_Z)
    {
      double rvec2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

      double D = rvec2 - new_Z*new_Z;

      if (D < D_min)
      {
        D_min = D;
        *next = neighbor;
        *dZ   = new_Z - Z;
      }
    }

  } /* end of n loop over neighbors */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
