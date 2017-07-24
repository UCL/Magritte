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



/* writing_output: write the output files
/*-----------------------------------------------------------------------------------------------*/

void write_output( double *unit_healpixvector, long *antipod,
                   GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                   double *pop, double *weight, double *energy )
{

  char   ch;

  long   n, n1, n2, r, i;                                                               /* index */

  int lspec;                                                 /* index for line producing species */


  /* Write the the grid again (only for debugging)  */

  FILE *outgrid = fopen("output/grid.txt", "w");

  if (outgrid == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (n=0; n<NGRID; n++){

    fprintf(outgrid, "%f\t%f\t%f\n", gridpoint[n].x, gridpoint[n].y, gridpoint[n].z);
  }

  fclose(outgrid);



  /* Write the unit HEALPix vectors */

  FILE *hp = fopen("output/healpix.txt", "w");

  if (hp == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


  for (r=0; r<NRAYS; r++){

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

  for (n1=0; n1<NGRID; n1++){

    for (n2=0; n2<NGRID; n2++){

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


  for (n1=0; n1<NGRID; n1++){

    for (n2=0; n2<NGRID; n2++){

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


  for (n=0; n<NGRID; n++){

    for (r=0; r<NRAYS; r++){

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


  for (n=0; n<NGRID; n++){

    for (r=0; r<NRAYS; r++){

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

  for (lspec=0; lspec<NLSPEC; lspec++){

  // fprintf(levelpops, "%d\t%d\t \n", lspec, nlev[lspec]);

    for (n=0; n<NGRID; n++){

      for (i=0; i<nlev[lspec]; i++){

        fprintf(levelpops, "%lE\t", pop[LSPECGRIDLEV(lspec,n, i)]);
      }

      fprintf(levelpops, "\n");
    }
  }

  fclose(levelpops);






}




/*-----------------------------------------------------------------------------------------------*/
