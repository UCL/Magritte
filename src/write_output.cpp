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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/grid" + tag + ".txt";

  FILE *outgrid = fopen(file_name.c_str(), "w");

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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/healpix" + tag + ".txt";

  FILE *hp = fopen(file_name.c_str(), "w");

  if (hp == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


  for (long r=0; r<NRAYS; r++){

    fprintf( hp, "%.15f\t%.15f\t%.15f\n", unit_healpixvector[VINDEX(r,0)],
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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/eval" + tag + ".txt";

  FILE *eval = fopen(file_name.c_str(), "w");

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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/key" + tag + ".txt";

  FILE *fkey = fopen(file_name.c_str(), "w");

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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/raytot" + tag + ".txt";

  FILE *rt = fopen(file_name.c_str(), "w");

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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/cum_raytot" + tag + ".txt";

  FILE *crt = fopen(file_name.c_str(), "w");

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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/abundances" + tag + ".txt";

  FILE *abun= fopen(file_name.c_str(), "w");

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

    name.erase(0,5);

    name.erase(name.end()-4,name.end());

    if( !tag.empty() ){

      tag = "_" + tag;
    }

    string file_name = "output/level_populations_" + name + tag + ".txt";

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

    name.erase(0,5);

    name.erase(name.end()-4,name.end());

    if( !tag.empty() ){

      tag = tag + "_" + tag;
    }

    string file_name = "output/line_intensities_" + name + tag + ".txt";

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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/temperature_gas" + tag + ".txt";

  FILE *temp_gas = fopen(file_name.c_str(), "w");

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


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/temperature_dust" + tag + ".txt";

  FILE *temp_dust = fopen(file_name.c_str(), "w");

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





/* write_UV_field: write the UV field at each point                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_UV_field(string tag, double *UV_field)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/UV_field" + tag + ".txt";

  FILE *UV_file = fopen(file_name.c_str(), "w");

  if (UV_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    fprintf( UV_file, "%lE\n", UV_field[n] );
  }


  fclose(UV_file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_UV_field: write the visual extinction (AV) at each point                                */
/*-----------------------------------------------------------------------------------------------*/

int write_AV(string tag, double *AV)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/AV" + tag + ".txt";

  FILE *AV_file = fopen(file_name.c_str(), "w");

  if (AV_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for(long r=0; r<NRAYS; r++){

      fprintf( AV_file, "%lE\t", AV[RINDEX(n,r)] );
    }

    fprintf( AV_file, "\n" );
  }


  fclose(AV_file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_rad_surface: write the rad surface at each point                                        */
/*-----------------------------------------------------------------------------------------------*/

int write_rad_surface(string tag, double *rad_surface)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  string file_name = "output/rad_surface" + tag + ".txt";

  FILE *rad_file = fopen(file_name.c_str(), "w");

  if (rad_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for(long r=0; r<NRAYS; r++){

      fprintf( rad_file, "%lE\t", rad_surface[RINDEX(n,r)] );
    }

    fprintf( rad_file, "\n" );
  }


  fclose(rad_file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
