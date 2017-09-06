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

#include <string>
#include <sstream>
using namespace std;

#include "declarations.hpp"
#include "write_output.hpp"



/* write_grid: write the grid again (for debugging)                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_grid(string tag, GRIDPOINT *gridpoint)
{


  FILE *outgrid = fopen("output/grid.txt", "w");

  if (outgrid == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (long n=0; n<NGRID; n++){

    fprintf(outgrid, "%f\t%f\t%f\n", gridpoint[n].x, gridpoint[n].y, gridpoint[n].z);
  }

  fclose(outgrid);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_healpixvectors: write the unit HEALPix vectors                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_healpixvectors(string tag, double *unit_healpixvector)
{


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


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_eval: Write the evaluation points (Z along ray and number of the ray)                   */
/*-----------------------------------------------------------------------------------------------*/

int write_eval(string tag, EVALPOINT *evalpoint)
{


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


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_key: write the key to find which grid point corresponds to which evaluation point       */
/*-----------------------------------------------------------------------------------------------*/

int write_key(string tag)
{


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


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_raytot: write the total of evaluation points along each ray                             */
/*-----------------------------------------------------------------------------------------------*/

int write_raytot(string tag)
{


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


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_cum_raytot: write the cumulative total of evaluation points along each ray              */
/*-----------------------------------------------------------------------------------------------*/

int write_cum_raytot(string tag)
{


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


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_abundances: write the abundances at each point                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_abundances(string tag)
{


  FILE *abun= fopen("output/abundances.txt", "w");

  if (abun == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (int spec=0; spec<NSPEC; spec++){

    for (long n=0; n<NGRID; n++){

      fprintf( abun, "%lE\t", species[spec].abn[n] );
    }

    fprintf( abun, "\n" );
  }


  fclose(abun);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_level_populations: write the level populations at each point for each transition        */
/*-----------------------------------------------------------------------------------------------*/

int write_level_populations(string tag, string *line_datafile, double *pop)
{


  for (int lspec=0; lspec<NLSPEC; lspec++){

    string name = line_datafile[lspec];

    string file_name = "output/level_populations_" + name.erase(0,5);

    FILE *levelpops = fopen(file_name.c_str(), "w");

    if (levelpops == NULL){

        printf("Error opening file!\n");
        exit(1);
      }

      for (long n=0; n<NGRID; n++){

        for (int i=0; i<nlev[lspec]; i++){

          fprintf(levelpops, "%lE\t", pop[LSPECGRIDLEV(lspec,n, i)]);
        }

        fprintf(levelpops, "\n");
      }

    fclose(levelpops);

  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_line_intensities: write the line intensities for each species, point and transition     */
/*-----------------------------------------------------------------------------------------------*/

int write_line_intensities(string tag, string *line_datafile, double *mean_intensity)
{


  for (int lspec=0; lspec<NLSPEC; lspec++){

    string name = line_datafile[lspec];

    string file_name = "output/line_intensities_" + name.erase(0,5);

    FILE *lintens = fopen(file_name.c_str(), "w");

    if (lintens == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

    for (int kr=0; kr<nrad[lspec]; kr++){

      for (long n=0; n<NGRID; n++){

        fprintf( lintens, "%lE\t", mean_intensity[LSPECGRIDRAD(lspec,n,kr)] );
      }

      fprintf( lintens, "\n" );
    }

    fclose(lintens);

  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_temperature_gas: write the gas temperatures at each point                               */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_gas(string tag, double *temperature_gas)
{


  FILE *temp_gas = fopen("output/temperature_gas.txt", "w");

  if (temp_gas == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    fprintf( temp_gas, "%lE\n", temperature_gas[n] );
  }


  fclose(temp_gas);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_temperature_dust: write the dust temperatures at each point                             */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_dust(string tag, double *temperature_dust)
{


  FILE *temp_dust = fopen("output/temperature_dust.txt", "w");

  if (temp_dust == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    fprintf( temp_dust, "%lE\n", temperature_dust[n] );
  }


  fclose(temp_dust);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
