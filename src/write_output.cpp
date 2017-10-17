/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Magritte: writing_output                                                                         */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <string>
#include <sstream>

#include "declarations.hpp"
#include "write_output.hpp"
#include "radfield_tools.hpp"



/* write_grid: write the grid again (for debugging)                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_grid(std::string tag, GRIDPOINT *gridpoint)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/grid" + tag + ".txt";

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

int write_healpixvectors(std::string tag, double *unit_healpixvector)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/healpix" + tag + ".txt";

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

int write_eval(std::string tag, EVALPOINT *evalpoint)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/eval" + tag + ".txt";

  FILE *eval = fopen(file_name.c_str(), "w");

  if (eval == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (long n1=0; n1<NGRID; n1++){

    for (long n2=0; n2<NGRID; n2++){

      fprintf( eval, "%lE\t%ld\t%d\n",
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

int write_key(std::string tag)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/key" + tag + ".txt";

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

int write_raytot(std::string tag)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/raytot" + tag + ".txt";

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

int write_cum_raytot(std::string tag)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/cum_raytot" + tag + ".txt";

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

int write_abundances(std::string tag)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/abundances" + tag + ".txt";

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

int write_level_populations(std::string tag, std::string *line_datafile, double *pop)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string name = line_datafile[lspec];

    name.erase(0,5);

    name.erase(name.end()-4,name.end());

    std::string file_name = "output/files/level_populations_" + name + tag + ".txt";

    FILE *levelpops = fopen(file_name.c_str(), "w");

    if (levelpops == NULL){

        std :: cout << "Error opening file " << file_name << "!\n";
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

int write_line_intensities(std::string tag, std::string *line_datafile, double *mean_intensity)
{


  if( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string name = line_datafile[lspec];

    name.erase(0,5);

    name.erase(name.end()-4,name.end());

    std::string file_name = "output/files/line_intensities_" + name + tag + ".txt";

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

int write_temperature_gas(std::string tag, double *temperature_gas)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/temperature_gas" + tag + ".txt";

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

int write_temperature_dust(std::string tag, double *temperature_dust)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/temperature_dust" + tag + ".txt";

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

int write_UV_field(std::string tag, double *UV_field)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/UV_field" + tag + ".txt";

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

int write_AV(std::string tag, double *AV)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/AV" + tag + ".txt";

  FILE *AV_file = fopen(file_name.c_str(), "w");

  if (AV_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

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

int write_rad_surface(std::string tag, double *rad_surface)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/rad_surface" + tag + ".txt";

  FILE *rad_file = fopen(file_name.c_str(), "w");

  if (rad_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf( rad_file, "%lE\t", rad_surface[RINDEX(n,r)] );
    }

    fprintf( rad_file, "\n" );
  }


  fclose(rad_file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_reaction_rates: write the rad surface at each point                                     */
/*-----------------------------------------------------------------------------------------------*/

int write_reaction_rates(std::string tag, REACTION *reaction)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/reaction_rates" + tag + ".txt";

  FILE *reac_file = fopen(file_name.c_str(), "w");

  if (reac_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (int reac=0; reac<NREAC; reac++){

      fprintf( reac_file, "%lE\t", reaction[reac].k[n] );
    }

    fprintf( reac_file, "\n" );
  }


  fclose(reac_file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_certain_reactions: write rates of certain reactions (as indicated in reaction_rates.cpp)*/
/*-----------------------------------------------------------------------------------------------*/

int write_certain_rates( std::string tag, std::string name, int nr_certain_reac,
                         int *certain_reactions, REACTION *reaction )
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }



  std::string file_name0 = "output/files/" + name + "_reactions" + tag + ".txt";

  FILE *certain_file0 = fopen(file_name0.c_str(), "w");

  if (certain_file0 == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (int reac=0; reac<nr_certain_reac; reac++){

    fprintf( certain_file0, "%d\n", certain_reactions[reac] );
  }

  fclose(certain_file0);



  std::string file_name = "output/files/" + name + "_rates" + tag + ".txt";

  FILE *certain_file = fopen(file_name.c_str(), "w");

  if (certain_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (int reac=0; reac<nr_certain_reac; reac++){

    fprintf( certain_file, "%d \t", certain_reactions[reac] );
  }


  fprintf( certain_file, "\n" );

  for (long n=0; n<NGRID; n++){

    for (int reac=0; reac<nr_certain_reac; reac++){

      fprintf( certain_file, "%lE \t", reaction[certain_reactions[reac]].k[n] );
    }

    fprintf( certain_file, "\n" );
  }


  fclose(certain_file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_double_1: write a 1D list of doubles                                                    */
/*-----------------------------------------------------------------------------------------------*/

int write_double_1(std::string name, std::string tag, long length, double *variable)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/" + name + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<length; n++){

    fprintf( file, "%lE\n", variable[n] );
  }


  fclose(file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_double_2: write a 2D array of doubles                                                   */
/*-----------------------------------------------------------------------------------------------*/

int write_double_2(std::string name, std::string tag, long nrows, long ncols, double *variable)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = "output/files/" + name + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long row=0; row<nrows; row++){

    for (long col=0; col<ncols; col++){

      fprintf( file, "%lE\t", variable[col + ncols*row] );
    }

    fprintf( file, "\n" );
  }


  fclose(file);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_radfield_tools: write the output of the functoins defined in radfield_tools             */
/*-----------------------------------------------------------------------------------------------*/

int write_radfield_tools( std::string tag, double *AV ,double lambda, double v_turb,
                          double *column_H2, double *column_CO )
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }



  /* Write dust scattering */

  std::string file_name = "output/files/dust_scattering" + tag + ".txt";

  FILE *ds_file = fopen(file_name.c_str(), "w");

  if (ds_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      double w = log10(1.0 + column_H2[RINDEX(n,r)]);
      double LLLlambda = (5675.0 - 200.6*w);

      fprintf( ds_file, "%lE\t", dust_scattering(AV[RINDEX(n,r)], LLLlambda) );
    }

    fprintf( ds_file, "\n" );
  }


  fclose(ds_file);



  /* Write H2 shield */

  std::string file_name2 = "output/files/shielding_H2" + tag + ".txt";

  FILE *s_file = fopen(file_name2.c_str(), "w");

  if (s_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }



  double doppler_width = v_turb / (lambda*1.0E-8);    /* linewidth (in Hz) of typical transition */
                                                /* (assuming turbulent broadening with b=3 km/s) */


  double radiation_width = 8.0E7;         /* radiative linewidth (in Hz) of a typical transition */


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf( s_file, "%lE\t", self_shielding_H2( column_H2[RINDEX(n,r)], doppler_width, radiation_width ) );
    }

    fprintf( s_file, "\n" );
  }


  fclose(s_file);


  /* Write CO shield */

  std::string file_name3 = "output/files/shielding_CO" + tag + ".txt";

  FILE *c_file = fopen(file_name3.c_str(), "w");

  if (c_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }



  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf( c_file, "%lE\t", self_shielding_CO( column_CO[RINDEX(n,r)], column_H2[RINDEX(n,r)] ) );
    }

    fprintf( c_file, "\n" );
  }


  fclose(c_file);



  /* Write X_lambda */

  std::string file_name4 = "output/files/X_lambda" + tag + ".txt";

  FILE *xl_file = fopen(file_name4.c_str(), "w");

  if (xl_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }



  for (long n=1; n<=200; n++){

    double LLLlambda = pow(10.0, (9.0-2.0)/200.0*n+2.0);
    fprintf( xl_file, "%lE\t%lE\n", LLLlambda, X_lambda(LLLlambda) );

  }


  fclose(xl_file);


  // cout << "X lambda " << X_lambda(1000.0) << "\n";

  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
