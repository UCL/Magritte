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
/*#include <mpi.h>*/

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "ray_tracing.hpp"
#include "heapsort.hpp"
#include "initializers.hpp"
#include "HEALPix/chealpix.hpp"




/* ray_tracing: creates the evaluation points for each ray for each grid point                   */
/*-----------------------------------------------------------------------------------------------*/

int ray_tracing( GRIDPOINT *gridpoint, EVALPOINT *evalpoint )
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

/*  MPI_Init(NULL, NULL);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (n1=rank; n1<NGRID; n1=n1+size){
*/


  #pragma omp parallel                                                                \
   shared( unit_healpixvector, gridpoint, evalpoint, key, raytot, cum_raytot, succes, \
           time_de, time_key, time_sort )                                             \
   default( none )
  {


  int num_threads = omp_get_num_threads();

  int thread_num  = omp_get_thread_num();


  long start = (thread_num*NGRID)/num_threads;

  long stop  = ((thread_num+1)*NGRID)/num_threads;  /* Note that the brackets are important here */



  for (long gridp=start; gridp<stop; gridp++){


    /* Place the origin at the location of the grid point under consideration */

    double origin[3];                   /* position vector of the grid point under consideration */

    origin[0] = gridpoint[gridp].x;
    origin[1] = gridpoint[gridp].y;
    origin[2] = gridpoint[gridp].z;


    /* Locate all grid points w.r.t. the origin */

    double ra2[NGRID];        /* array with the squares of the lengths of local position vectors */

    long   rb[NGRID];                /* array with the identifiers of the local position vectors */


    for (long n=0; n<NGRID; n++){

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

    double radius = sqrt( ra2[NGRID-1]  );

    time_sort += omp_get_wtime();


    long   nr[NRAYS];                      /* current number of evaluation points along each ray */

    initialize_long_array(nr, NRAYS);

    double Z[NRAYS];                                                       /* distance along ray */

    initialize_double_array(Z, NRAYS);


    /*   FIND EVALUATION POINTS FRO gridp                                                        */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Devide the grid points over the rays through the origin */
    /* Start from the second point in rb (first point is gridpoint itself) */

    time_de -= omp_get_wtime();


    for (long n=1; n<NGRID; n++){

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

      double rvec_dot_uhpv = rvec[0]*unit_healpixvector[VINDEX(ipix,0)]
	                           + rvec[1]*unit_healpixvector[VINDEX(ipix,1)]
	                           + rvec[2]*unit_healpixvector[VINDEX(ipix,2)];

      double cosine = (rvec_dot_uhpv - Z[ipix])
		                  / sqrt(ra2[n] - 2*Z[ipix]*rvec_dot_uhpv + Z[ipix]*Z[ipix]);


      /* Avoid nan angles because of rounding errors */

      if(cosine>1.0){

        cosine = 1.0;
      }


      double angle = acos( cosine );


      /* If angle < THETA_CRIT, add the new evaluation point */

      if (angle < THETA_CRIT){

        evalpoint[GINDEX(gridp,rb[n])].dZ  = rvec_dot_uhpv - Z[ipix];

        evalpoint[GINDEX(gridp,rb[n])].vol
          = (gridpoint[rb[n]].vx - gridpoint[gridp].vx)*unit_healpixvector[VINDEX(ipix,0)]
            + (gridpoint[rb[n]].vy - gridpoint[gridp].vy)*unit_healpixvector[VINDEX(ipix,1)]
            + (gridpoint[rb[n]].vz - gridpoint[gridp].vz)*unit_healpixvector[VINDEX(ipix,2)];

        evalpoint[GINDEX(gridp,rb[n])].ray = ipix;

        raytot[RINDEX(gridp,ipix)] = raytot[RINDEX(gridp,ipix)] + 1;

        Z[ipix] = rvec_dot_uhpv;

	      evalpoint[GINDEX(gridp,rb[n])].Z = Z[ipix];

        evalpoint[GINDEX(gridp,rb[n])].onray = true;

        succes = succes + 1;


        /* Check whether ipix ray for evaluation point can be considered equivalent */

        double distance_to_ray2 = ra2[n] - rvec_dot_uhpv * rvec_dot_uhpv;

        if (distance_to_ray2 < RAY_SEPARATION2){

          evalpoint[GINDEX(gridp,rb[n])].eqp = gridp;
        }

        else {

          evalpoint[GINDEX(gridp,rb[n])].eqp = rb[n];
        }

      } /* end of if angle < THETA_CRIT */

    } /* end of n loop over gridpoints (around an origin) */


    time_de += omp_get_wtime();


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   SETUP EVALPOINTS DATA STRUCTURE                                                         */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    cum_raytot[RINDEX(gridp,0)] = 0;

    cum_raytot[RINDEX(gridp,1)] = raytot[RINDEX(gridp,0)];


    for (long r=2; r<NRAYS; r++){

      cum_raytot[RINDEX(gridp,r)] = cum_raytot[RINDEX(gridp,r-1)] + raytot[RINDEX(gridp,r-1)];
    }


    /* Make a key to find back which evaluation point is where on which ray */

    time_key -= omp_get_wtime();

    for (long n=0; n<NGRID; n++){

      if (evalpoint[GINDEX(gridp,rb[n])].onray == true){

        long ray = evalpoint[GINDEX(gridp,rb[n])].ray;

        GP_NR_OF_EVALP(gridp, ray, nr[ray]) = rb[n];

        nr[ray] = nr[ray] + 1;
      }

    }

    time_key += omp_get_wtime();


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  } /* end of gridp loop over gridpoints (origins) */
  } /* end of OpenMP parallel region */


  printf( "(ray_tracing): time in dividing evaluation points %lf sec\n", time_de );
  printf( "(ray_tracing): time in making the key             %lf sec\n", time_key );
  printf( "(ray_tracing): time in heapsort routine           %lf sec\n", time_sort );

  printf( "(ray_tracing): succes rate %.2lf" \
          "(# eval. points)/ ((# grid points)^2 - (# grid points)) \n",
          (double) succes/(NGRID*NGRID-NGRID) );


/*
  if (rank != 0){

    MPI_Send( void* data, int count, MPI_Datatype datatype, int destination, int tag,
              MPI_Comm communicator );
  }
  else {

    MPI_Recv( void* data, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm communicator, MPI_Status* status );
  }
*/
/*  MPI_Finalize(); */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* get_evalpoints: creates the evaluation points for each ray for this grid point                */
/*-----------------------------------------------------------------------------------------------*/

int get_evalpoints( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long gridp )
{


  /* Initialize the data structures that will store the evaluation points */

  initialize_long_array(local_key, NGRID);

  initialize_long_array(local_raytot, NRAYS);

  initialize_long_array(local_cum_raytot, NRAYS);


  /* Place the origin at the location of the grid point under consideration */

  double origin[3];                     /* position vector of the grid point under consideration */

  origin[0] = gridpoint[gridp].x;
  origin[1] = gridpoint[gridp].y;
  origin[2] = gridpoint[gridp].z;


  /* Locate all grid points w.r.t. the origin */

  double ra2[NGRID];          /* array with the squares of the lengths of local position vectors */

  long rb[NGRID];                    /* array with the identifiers of the local position vectors */


  for (long n=0; n<NGRID; n++){

    double rvec[3];                       /* local position vector of a grid point w.r.t. origin */

    rvec[0] = gridpoint[n].x - origin[0];
    rvec[1] = gridpoint[n].y - origin[1];
    rvec[2] = gridpoint[n].z - origin[2];

    ra2[n] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];              /* SQUARE length! */
    rb[n]  = n;
  }


  /* Sort the grid points w.r.t their distance from the origin */

  heapsort(ra2, rb, NGRID);

  double radius = sqrt( ra2[NGRID-1]  );


  double Z[NRAYS];                                                         /* distance along ray */

  initialize_double_array(Z, NRAYS);


  /*   FIND EVALUATION POINTS FOR gridp                                                          */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  /* Devide the grid points over the rays through the origin */
  /* Start from the second point in rb (first point is gridpoint itself) */


  for (long n=1; n<NGRID; n++){

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

    double rvec_dot_uhpv = rvec[0]*unit_healpixvector[VINDEX(ipix,0)]
	                         + rvec[1]*unit_healpixvector[VINDEX(ipix,1)]
	                         + rvec[2]*unit_healpixvector[VINDEX(ipix,2)];

    double cosine = (rvec_dot_uhpv - Z[ipix])
	                  / sqrt(ra2[n] - 2*Z[ipix]*rvec_dot_uhpv + Z[ipix]*Z[ipix]);


    /* Avoid nan angles because of rounding errors */

    if(cosine>1.0){

      cosine = 1.0;
    }


    double angle = acos( cosine );


    /* If angle < THETA_CRIT, add the new evaluation point */

    if (angle < THETA_CRIT){

      evalpoint[rb[n]].onray = true;

      evalpoint[rb[n]].dZ    = rvec_dot_uhpv - Z[ipix];

      evalpoint[rb[n]].ray   = ipix;

	    evalpoint[rb[n]].Z     = Z[ipix] = rvec_dot_uhpv;

      local_raytot[ipix] = local_raytot[ipix] + 1;


      /* Check whether ipix ray for evaluation point can be considered equivalent */

      double distance_to_ray2 = ra2[n] - rvec_dot_uhpv * rvec_dot_uhpv;

      if (distance_to_ray2 < RAY_SEPARATION2){

        evalpoint[rb[n]].eqp = gridp;
      }

      else {

        evalpoint[rb[n]].eqp = rb[n];
      }


    } /* end of if angle < THETA_CRIT */

  } /* end of n loop over gridpoints (around an origin) */


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /*   SETUP EVALPOINTS DATA STRUCTURE                                                           */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  long nr[NRAYS];                              /* current number of evaluation points on the ray */

  initialize_long_array(nr, NRAYS);


  local_cum_raytot[0] = 0;


  for (long r=1; r<NRAYS; r++){

    local_cum_raytot[r] = local_cum_raytot[r-1] + local_raytot[r-1];
  }


  /* Make a key to find back which evaluation point is where on which ray */

  for (long n=0; n<NGRID; n++){

    if (evalpoint[rb[n]].onray == true){

      long ray = evalpoint[rb[n]].ray;

      LOCAL_GP_NR_OF_EVALP(ray, nr[ray]) = rb[n];

      nr[ray] = nr[ray] + 1;
    }

  }


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
