/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* 3D-RT: writing_output                                                                         */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>

#include "declarations.hpp"
#include "write_output.hpp"



/* writing_output: write the output files
/*-----------------------------------------------------------------------------------------------*/

void write_output( double *unit_healpixvector, long *antipod,
                   GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                   double *pop, double *weight, double *energy )
{


  /* Write the the grid again (only for debugging)  */

  FILE *outgrid = fopen("output/grid.txt", "w");

  if (outgrid == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (long n=0; n<NGRID; n++){

    fprintf(outgrid, "%f\t%f\t%f\n", gridpoint[n].x, gridpoint[n].y, gridpoint[n].z);
  }

  fclose(outgrid);



  /* Write the unit HEALPix vectors */

  FILE *hp = fopen("output/healpix.txt", "w");

  if (hp == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


  for (long r=0; r<NRAYS; r++){

    fprintf( hp, "%f\t%f\t%f\n", unit_healpixvector[VINDEX(r,0)],
                                 unit_healpixvector[VINDEX(r,1)],
                                 unit_healpixvector[VINDEX(r,2)] );
  }

  fclose(hp);



  /* Write the evaluation points (Z along ray and number of the ray) */

  FILE *eval = fopen("output/eval.txt", "w");

  if (eval == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (long n1=0; n1<NGRID; n1++){

    for (long n2=0; n2<NGRID; n2++){

      fprintf( eval, "%lf\t%ld\t%d\n",
               evalpoint[GINDEX(n1,n2)].Z,
               evalpoint[GINDEX(n1,n2)].ray,
               evalpoint[GINDEX(n1,n2)].onray );
    }

  }

  fclose(eval);



  /* Write the key to find which grid point corresponds to which evaluation point */

  FILE *fkey = fopen("output/key.txt", "w");

  if (fkey == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


  for (long n1=0; n1<NGRID; n1++){

    for (long n2=0; n2<NGRID; n2++){

      fprintf(fkey, "%ld\t", key[GINDEX(n1,n2)] );
    }

    fprintf(fkey, "\n");
  }

  fclose(fkey);



  /* Write the total of evaluation points along each ray */

  FILE *rt = fopen("output/raytot.txt", "w");

  if (rt == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf(rt, "%ld\t", raytot[RINDEX(n,r)] );
    }

    fprintf(rt, "\n");
  }

  fclose(rt);



  /* Write the cumulative total of evaluation points along each ray */

  FILE *crt = fopen("output/cum_raytot.txt", "w");

  if (crt == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf(crt, "%ld\t", cum_raytot[RINDEX(n,r)] );
    }

    fprintf(crt, "\n");
  }

  fclose(crt);



  /* Write the level populations */

  FILE *levelpops = fopen("output/level_populations.txt", "w");

  if (levelpops == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  fprintf(levelpops, "%d\n", NLSPEC);

  for (int lspec=0; lspec<NLSPEC; lspec++){

  // fprintf(levelpops, "%d\t%d\t \n", lspec, nlev[lspec]);

    for (long n=0; n<NGRID; n++){

      for (int i=0; i<nlev[lspec]; i++){

        fprintf(levelpops, "%lE\t", pop[LSPECGRIDLEV(lspec,n, i)]);
      }

      fprintf(levelpops, "\n");
    }
  }

  fclose(levelpops);






}




/*-----------------------------------------------------------------------------------------------*/
