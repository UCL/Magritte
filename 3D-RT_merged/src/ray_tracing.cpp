/* Frederik De Ceuster - University College London                                               */
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

#include "heapsort.cpp"
#include "HEALPix/vec2ang.c"
#include "HEALPix/ang2pix_nest.c"



/* ray_tracing: creates the evaluation points for each ray for each grid point                   */
/*-----------------------------------------------------------------------------------------------*/

void ray_tracing( double theta_crit, double ray_separation2, double *unit_healpixvector,
                  GRIDPOINT *gridpoint, EVALPOINT *evalpoint )
{

  long   n;                                                                             /* index */
  long   n1, n2, n3, n4;                                                   /* grid point indices */
  long   r1, r2, r3, r4, r5, r6, ray;                                             /* ray indices */
  long   e1;                                                           /* evaluation point index */
  long   ipix;                                          /* ray index (as reference to the pixel) */
  long   rb[NGRID];                  /* array with the identifiers of the local position vectors */
  long   nr[NRAYS];                        /* current number of evaluation points along each ray */
  long   succes=0;                                               /* sum of all evaluation points */
  long   start, stop;                                     /* delimiters in for loops with openmp */

  double vector[3];                        /* unit vector in the direction of the HEALPix vector */
  double vp;                                             /* grid point velocity projected on ray */
  double origin[3];                     /* position vector of the grid point under consideration */
  double rvec[3];                         /* local position vector of a grid point w.r.t. origin */
  double ra2[NGRID];          /* array with the squares of the lengths of local position vectors */
  double radius;                                                 /* maximal distance from origin */
  double theta, phi;                                                /* angles of the HEALPix ray */
  double healpixvector[3];                         /* scaled HEALPix vector representing the ray */
  double rvec_dot_uhpv;                            /* dot product of rvec and unit_healpixvector */
  double angle;        /* angle between gridpoint and ray as seen from previous evaluation point */
  double Z[NRAYS];                                                         /* distance along ray */

  double distance_to_ray2;    /* square distance between grid p. and corresponding evaluation p. */

  double time_de=0.0;                                      /* time in dividing evaluation points */
  double time_key=0.0;                                                   /* time to make the key */
  double time_sort=0.0;                  /* time to sort grid points w.r.t. distance from origin */


  /* Note: GINDEX, RINDEX and GP_NR_OF_EVALP are defined in definitions.h */



  printf("(ray_tracing): number of grid points     %*d\n", MAX_WIDTH, NGRID);
  printf("(ray_tracing): number of HEALPix sides   %*d\n", MAX_WIDTH, NSIDES);
  printf("(ray_tracing): number of HEALPix rays    %*d\n", MAX_WIDTH, NRAYS);
  printf("(ray_tracing): critical angle theta_crit %*.2lf\n", MAX_WIDTH, theta_crit);
  printf("(ray_tracing): equivalent ray distance   %*.2lE\n", MAX_WIDTH, ray_separation2);




  /* For all grid points */

/*  MPI_Init(NULL, NULL);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (n1=rank; n1<NGRID; n1=n1+size){
*/


  #pragma omp parallel \
   private( n1, n2, n3, n4, ra2, rb, origin, rvec, theta, phi, radius, ipix, rvec_dot_uhpv, angle, \
            vp, r5, r6, start, stop, healpixvector, distance_to_ray2, Z, nr, ray ) \
   shared( gridpoint, unit_healpixvector, key, \
           evalpoint, raytot, succes, ray_separation2, theta_crit, time_sort, time_de, cum_raytot, time_key ) \
   default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  start = (thread_num*NGRID)/num_threads;            
  stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note that the brackets are important here */



  for (n1=start; n1<stop; n1++){


    /* Place the origin at the location of the grid point under consideration */

    origin[0] = gridpoint[n1].x;
    origin[1] = gridpoint[n1].y;
    origin[2] = gridpoint[n1].z;


    /* Locate all grid points w.r.t. the origin */

    for (n2=0; n2<NGRID; n2++){

      rvec[0] = gridpoint[n2].x - origin[0];
      rvec[1] = gridpoint[n2].y - origin[1];
      rvec[2] = gridpoint[n2].z - origin[2];

      ra2[n2] = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];           /* SQUARE length! */
      rb[n2]  = n2;
    }



    /* Sort the grid points w.r.t their distance from the origin */

    time_sort -= omp_get_wtime();

    heapsort(ra2, rb, NGRID);

    radius = sqrt( ra2[NGRID-1]  );

    time_sort += omp_get_wtime();



    // UNNECESSARY AND CONFUSING !

    // /* The first evaluation point for all rays is the origin */

    // evalpoint[GINDEX(n1,0)].dZ = 0.0;
    // evalpoint[GINDEX(n1,0)].ray = 0;
    // evalpoint[GINDEX(n1,n1)].nr = 0;



    for (r5=0; r5<NRAYS; r5++){

      nr[r5] = 0;                     /* current nr. of evaluation points on the ray: the origin */

      Z[r5] = 0.0;                      /* current distance from last evaluation point to origin */

      raytot[RINDEX(n1,r5)] = 0;
    }


    /* Devide the grid points over the rays through the origin */
    /* Start from the second point in rb (first point is gridpoint itself) */

    time_de -= omp_get_wtime();


    for (n3=1; n3<NGRID; n3++){

      rvec[0] = gridpoint[rb[n3]].x - origin[0];
      rvec[1] = gridpoint[rb[n3]].y - origin[1];
      rvec[2] = gridpoint[rb[n3]].z - origin[2];


      /* Get ipix ray where rvec belongs to (using HEALPix functions) */

      void vec2ang(double *rvec, double *theta, double *phi);

      vec2ang(rvec, &theta, &phi);


      void ang2pix_nest( const long nside, double theta, double phi, long *ipix);

      ang2pix_nest(NSIDES, theta, phi, &ipix);

/*
      printf("(ray-tracing): %ld looks at %ld with ipix %ld\n", n1, n3, ipix);
*/

      /* Create the corresponding healpixvector for each ipix pixel */

      healpixvector[0] = 1.1*radius*unit_healpixvector[VINDEX(ipix,0)];
      healpixvector[1] = 1.1*radius*unit_healpixvector[VINDEX(ipix,1)];
      healpixvector[2] = 1.1*radius*unit_healpixvector[VINDEX(ipix,2)];


      /* Calculate the angle between the gridpoint and its corresponding ray */

      rvec_dot_uhpv = rvec[0]*unit_healpixvector[VINDEX(ipix,0)]
	                    + rvec[1]*unit_healpixvector[VINDEX(ipix,1)]
	                    + rvec[2]*unit_healpixvector[VINDEX(ipix,2)];

      angle = acos( (rvec_dot_uhpv - Z[ipix])
		                / sqrt(ra2[n3] - 2*Z[ipix]*rvec_dot_uhpv + Z[ipix]*Z[ipix]) );
/*
      printf("(ray-tracing): angle %lf\n", angle);
*/

      /* If angle < theta_crit add the new evaluation point */

      if (angle < theta_crit){

        vp = (gridpoint[rb[n3]].vx - gridpoint[n1].vx)*unit_healpixvector[VINDEX(ipix,0)]
             + (gridpoint[rb[n3]].vy - gridpoint[n1].vy)*unit_healpixvector[VINDEX(ipix,1)]
	           + (gridpoint[rb[n3]].vz - gridpoint[n1].vz)*unit_healpixvector[VINDEX(ipix,2)];


        evalpoint[GINDEX(n1,rb[n3])].dZ  = rvec_dot_uhpv - Z[ipix];
        evalpoint[GINDEX(n1,rb[n3])].vol = evalpoint[GINDEX(n1,rb[n3])].vol - vp;
        evalpoint[GINDEX(n1,rb[n3])].ray = ipix;

        raytot[RINDEX(n1,ipix)] = raytot[RINDEX(n1,ipix)] + 1;

        Z[ipix] = rvec_dot_uhpv;

	      evalpoint[GINDEX(n1,rb[n3])].Z = Z[ipix];

        evalpoint[GINDEX(n1,rb[n3])].onray = true;
        succes = succes + 1;


        /* Check whether ipix ray for evaluation point can be considered equivalent */

        distance_to_ray2 = ra2[n3] - rvec_dot_uhpv * rvec_dot_uhpv;

        if (distance_to_ray2 < ray_separation2){

          evalpoint[GINDEX(n1,rb[n3])].eqp = n1;
        }
        else {

          evalpoint[GINDEX(n1,rb[n3])].eqp = rb[n3];
        }

        // printf("(ray_tracing): eqp %ld\n", evalpoint[GINDEX(n1,rb[n3])].eqp);
      }


    } /* end of n3 loop over gridpoints (around an origin) */


    time_de += omp_get_wtime();


    cum_raytot[RINDEX(n1,0)] = 0;
    cum_raytot[RINDEX(n1,1)] = raytot[RINDEX(n1,0)];


    for (r6=2; r6<NRAYS; r6++){

      cum_raytot[RINDEX(n1,r6)] = cum_raytot[RINDEX(n1,r6-1)] + raytot[RINDEX(n1,r6-1)];
    }


    /* Make a key to find back which evaluation point is where on which ray */

    time_key -= omp_get_wtime();

    for (n4=0; n4<NGRID; n4++){

      if (evalpoint[GINDEX(n1,rb[n4])].onray == true){

        ray = evalpoint[GINDEX(n1,rb[n4])].ray;

        GP_NR_OF_EVALP(n1, ray, nr[ray]) = rb[n4];

        nr[ray] = nr[ray] + 1;
      }

    }

    time_key += omp_get_wtime();

  } /* end of n1 loop over gridpoints (origins) */
  } /* end of OpenMP parallel region */


  printf("(ray_tracing): time in dividing evaluation points %lf sec\n", time_de);
  printf("(ray_tracing): time in making the key             %lf sec\n", time_key);
  printf("(ray_tracing): time in heapsort routine           %lf sec\n", time_sort);

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

}

/*-----------------------------------------------------------------------------------------------*/
