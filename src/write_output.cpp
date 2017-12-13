/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Magritte: writing_output                                                                      */
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

#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>

#include <vtkCellData.h>
#include <vtkVersion.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "write_output.hpp"
#include "species_tools.hpp"
#include "radfield_tools.hpp"
#include "initializers.hpp"
#include "calc_LTE_populations.hpp"
#include "../setup/outputdirectory.hpp"



/* write_grid: write the grid again (for debugging)                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_grid(std::string tag, GRIDPOINT *gridpoint)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "grid" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }

  for (long n=0; n<NGRID; n++){

    fprintf(file, "%f\t%f\t%f\n", gridpoint[n].x, gridpoint[n].y, gridpoint[n].z);
  }

  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_healpixvectors: write the unit HEALPix vectors                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_healpixvectors(std::string tag)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "healpix" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long r=0; r<NRAYS; r++){

    fprintf(file, "%.15f\t%.15f\t%.15f\n", unit_healpixvector[VINDEX(r,0)],
                                           unit_healpixvector[VINDEX(r,1)],
                                           unit_healpixvector[VINDEX(r,2)] );
  }

  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





#if !( ON_THE_FLY )

/* write_eval: Write the evaluation points (Z along ray and number of the ray)                   */
/*-----------------------------------------------------------------------------------------------*/

int write_eval(std::string tag, EVALPOINT *evalpoint)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "eval" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }

  for (long n1=0; n1<NGRID; n1++){

    for (long n2=0; n2<NGRID; n2++){

      fprintf( file, "%lE\t%ld\t%d\n",
               evalpoint[GINDEX(n1,n2)].Z,
               evalpoint[GINDEX(n1,n2)].ray,
               evalpoint[GINDEX(n1,n2)].onray );
    }

  }

  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_key: write the key to find which grid point corresponds to which evaluation point       */
/*-----------------------------------------------------------------------------------------------*/

int write_key(std::string tag, long *key)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "key" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n1=0; n1<NGRID; n1++){

    for (long n2=0; n2<NGRID; n2++){

      fprintf(file, "%ld\t", key[GINDEX(n1,n2)] );
    }

    fprintf(file, "\n");
  }

  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_raytot: write the total of evaluation points along each ray                             */
/*-----------------------------------------------------------------------------------------------*/

int write_raytot(std::string tag, long *raytot)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "raytot" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf(file, "%ld\t", raytot[RINDEX(n,r)] );
    }

    fprintf(file, "\n");
  }

  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_cum_raytot: write the cumulative total of evaluation points along each ray              */
/*-----------------------------------------------------------------------------------------------*/

int write_cum_raytot(std::string tag, long *cum_raytot)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "cum_raytot" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf(file, "%ld\t", cum_raytot[RINDEX(n,r)] );
    }

    fprintf(file, "\n");
  }

  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/

#endif





/* write_abundances: write the abundances at each point                                          */
/*-----------------------------------------------------------------------------------------------*/

int write_abundances(std::string tag)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "abundances" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (int spec=0; spec<NSPEC; spec++){

      fprintf( file, "%lE\t", species[spec].abn[n] );
    }

    fprintf( file, "\n" );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_level_populations: write the level populations at each point for each transition        */
/*-----------------------------------------------------------------------------------------------*/

int write_level_populations(std::string tag, double *pop)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string lspec_name = species[ lspec_nr[lspec] ].sym;

    std::string file_name = OUTPUT_DIRECTORY + "level_populations_" + lspec_name + tag + ".txt";

    FILE *file = fopen(file_name.c_str(), "w");


    if (file == NULL){

      std :: cout << "Error opening file " << file_name << "!\n";
      std::cout << file_name + "\n";
      exit(1);
    }


    for (long n=0; n<NGRID; n++){

      for (int i=0; i<nlev[lspec]; i++){

        fprintf(file, "%lE\t", pop[LSPECGRIDLEV(lspec,n, i)]);
      }

      fprintf(file, "\n");
    }


    fclose(file);

  } /* end of lspec loop over line producing species */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_line_intensities: write the line intensities for each species, point and transition     */
/*-----------------------------------------------------------------------------------------------*/

int write_line_intensities(std::string tag, double *mean_intensity)
{


  if( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string lspec_name = species[ lspec_nr[lspec] ].sym;

    std::string file_name = OUTPUT_DIRECTORY + "line_intensities_" + lspec_name + tag + ".txt";

    FILE *file = fopen(file_name.c_str(), "w");


    if (file == NULL){

      printf("Error opening file!\n");
      std::cout << file_name + "\n";
      exit(1);
    }


    for (long n=0; n<NGRID; n++){

      for (int kr=0; kr<nrad[lspec]; kr++){

        fprintf( file, "%lE\t", mean_intensity[LSPECGRIDRAD(lspec,n,kr)] );
      }

      fprintf( file, "\n" );
    }

    fclose(file);

  }


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_temperature_gas: write the gas temperatures at each point                               */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_gas(std::string tag, double *temperature_gas)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "temperature_gas" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    fprintf( file, "%lE\n", temperature_gas[n] );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_temperature_dust: write the dust temperatures at each point                             */
/*-----------------------------------------------------------------------------------------------*/

int write_temperature_dust(std::string tag, double *temperature_dust)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "temperature_dust" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    fprintf( file, "%lE\n", temperature_dust[n] );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_UV_field: write the UV field at each point                                              */
/*-----------------------------------------------------------------------------------------------*/

int write_UV_field(std::string tag, double *UV_field)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "UV_field" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    fprintf( file, "%lE\n", UV_field[n] );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_UV_field: write the visual extinction (AV) at each point                                */
/*-----------------------------------------------------------------------------------------------*/

int write_AV(std::string tag, double *AV)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "AV" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf( file, "%lE\t", AV[RINDEX(n,r)] );
    }

    fprintf( file, "\n" );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_rad_surface: write the rad surface at each point                                        */
/*-----------------------------------------------------------------------------------------------*/

int write_rad_surface(std::string tag, double *rad_surface)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "rad_surface" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      fprintf( file, "%lE\t", rad_surface[RINDEX(n,r)] );
    }

    fprintf( file, "\n" );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_reaction_rates: write the rad surface at each point                                     */
/*-----------------------------------------------------------------------------------------------*/

int write_reaction_rates(std::string tag, REACTION *reaction)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + "reaction_rates" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<NGRID; n++){

    for (int reac=0; reac<NREAC; reac++){

      fprintf( file, "%lE\t", reaction[reac].k[n] );
    }

    fprintf( file, "\n" );
  }


  fclose(file);


  return EXIT_SUCCESS;

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



  std::string file_name0 = OUTPUT_DIRECTORY + name + "_reactions" + tag + ".txt";

  FILE *file0 = fopen(file_name0.c_str(), "w");

  if (file0 == NULL){

    printf("Error opening file!\n");
    std::cout << file_name0 + "\n";
    exit(1);
  }


  for (int reac=0; reac<nr_certain_reac; reac++){

    fprintf( file0, "%d\n", certain_reactions[reac] );
  }

  fclose(file0);



  std::string file_name = OUTPUT_DIRECTORY + name + "_rates" + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (int reac=0; reac<nr_certain_reac; reac++){

    fprintf( file, "%d \t", certain_reactions[reac] );
  }


  fprintf( file, "\n" );

  for (long n=0; n<NGRID; n++){

    for (int reac=0; reac<nr_certain_reac; reac++){

      fprintf( file, "%lE \t", reaction[certain_reactions[reac]].k[n] );
    }

    fprintf( file, "\n" );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_double_1: write a 1D list of doubles                                                    */
/*-----------------------------------------------------------------------------------------------*/

int write_double_1(std::string name, std::string tag, long length, double *variable)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + name + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long n=0; n<length; n++){

    fprintf( file, "%lE\n", variable[n] );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_double_2: write a 2D array of doubles                                                   */
/*-----------------------------------------------------------------------------------------------*/

int write_double_2(std::string name, std::string tag, long nrows, long ncols, double *variable)
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }

  std::string file_name = OUTPUT_DIRECTORY + name + tag + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");

  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }


  for (long row=0; row<nrows; row++){

    for (long col=0; col<ncols; col++){

      fprintf( file, "%lE\t", variable[col + ncols*row] );
    }

    fprintf( file, "\n" );
  }


  fclose(file);


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_radfield_tools: write the output of the functoins defined in radfield_tools             */
/*-----------------------------------------------------------------------------------------------*/
//
// int write_radfield_tools( std::string tag, double *AV ,double lambda,
//                           double *column_H2, double *column_CO )
// {
//
//
//   if ( !tag.empty() ){
//
//     tag = "_" + tag;
//   }
//
//
//
//   /* Write dust scattering */
//
//   std::string file_name = "output/files/dust_scattering" + tag + ".txt";
//
//   FILE *ds_file = fopen(file_name.c_str(), "w");
//
//   if (ds_file == NULL){
//
//     printf("Error opening file!\n");
//     exit(1);
//   }
//
//
//   for (long n=0; n<NGRID; n++){
//
//     for (long r=0; r<NRAYS; r++){
//
//       double w = log10(1.0 + column_H2[RINDEX(n,r)]);
//       double LLLlambda = (5675.0 - 200.6*w);
//
//       fprintf( ds_file, "%lE\t", dust_scattering(AV[RINDEX(n,r)], LLLlambda) );
//     }
//
//     fprintf( ds_file, "\n" );
//   }
//
//
//   fclose(ds_file);
//
//
//
//   /* Write H2 shield */
//
//   std::string file_name2 = "output/files/shielding_H2" + tag + ".txt";
//
//   FILE *s_file = fopen(file_name2.c_str(), "w");
//
//   if (s_file == NULL){
//
//     printf("Error opening file!\n");
//     exit(1);
//   }
//
//
//
//   double doppler_width = V_TURB / (lambda*1.0E-8);    /* linewidth (in Hz) of typical transition */
//                                                 /* (assuming turbulent broadening with b=3 km/s) */
//
//
//   double radiation_width = 8.0E7;         /* radiative linewidth (in Hz) of a typical transition */
//
//
//   for (long n=0; n<NGRID; n++){
//
//     for (long r=0; r<NRAYS; r++){
//
//       fprintf( s_file, "%lE\t", self_shielding_H2( column_H2[RINDEX(n,r)], doppler_width, radiation_width ) );
//     }
//
//     fprintf( s_file, "\n" );
//   }
//
//
//   fclose(s_file);
//
//
//   /* Write CO shield */
//
//   std::string file_name3 = "output/files/shielding_CO" + tag + ".txt";
//
//   FILE *c_file = fopen(file_name3.c_str(), "w");
//
//   if (c_file == NULL){
//
//     printf("Error opening file!\n");
//     exit(1);
//   }
//
//
//
//   for (long n=0; n<NGRID; n++){
//
//     for (long r=0; r<NRAYS; r++){
//
//       fprintf( c_file, "%lE\t", self_shielding_CO( column_CO[RINDEX(n,r)], column_H2[RINDEX(n,r)] ) );
//     }
//
//     fprintf( c_file, "\n" );
//   }
//
//
//   fclose(c_file);
//
//
//
//   /* Write X_lambda */
//
//   std::string file_name4 = "output/files/X_lambda" + tag + ".txt";
//
//   FILE *xl_file = fopen(file_name4.c_str(), "w");
//
//   if (xl_file == NULL){
//
//     printf("Error opening file!\n");
//     exit(1);
//   }
//
//
//
//   for (long n=1; n<=200; n++){
//
//     double LLLlambda = pow(10.0, (9.0-2.0)/200.0*n+2.0);
//     fprintf( xl_file, "%lE\t%lE\n", LLLlambda, X_lambda(LLLlambda) );
//
//   }
//
//
//   fclose(xl_file);
//
//
//   // cout << "X lambda " << X_lambda(1000.0) << "\n";
//
//   return EXIT_SUCCESS;
//
// }

/*-----------------------------------------------------------------------------------------------*/





/* write_Einstein_coeff: write the Einstein A, B or C coefficients                               */
/*-----------------------------------------------------------------------------------------------*/

int write_Einstein_coeff( std::string tag, double *A_coeff, double *B_coeff, double *C_coeff )
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string lspec_name = species[ lspec_nr[lspec] ].sym;


    std::string file_name_A = OUTPUT_DIRECTORY + "Einstein_A_" + lspec_name + tag + ".txt";
    std::string file_name_B = OUTPUT_DIRECTORY + "Einstein_B_" + lspec_name + tag + ".txt";
    std::string file_name_C = OUTPUT_DIRECTORY + "Einstein_C_" + lspec_name + tag + ".txt";


    FILE *file_A = fopen(file_name_A.c_str(), "w");
    FILE *file_B = fopen(file_name_B.c_str(), "w");
    FILE *file_C = fopen(file_name_C.c_str(), "w");

    if (file_A == NULL){

      printf("Error opening file!\n");
      std::cout << file_name_A + "\n";
      exit(1);
    }

    if (file_B == NULL){

      printf("Error opening file!\n");
      std::cout << file_name_B + "\n";
      exit(1);
    }

    if (file_C == NULL){

      printf("Error opening file!\n");
      std::cout << file_name_C + "\n";
      exit(1);
    }


    for (long row=0; row<nlev[lspec]; row++){

      for (long col=0; col<nlev[lspec]; col++){

        fprintf( file_A, "%lE\t", A_coeff[LSPECLEVLEV(lspec,row,col)] );
        fprintf( file_B, "%lE\t", B_coeff[LSPECLEVLEV(lspec,row,col)] );
        fprintf( file_C, "%lE\t", C_coeff[LSPECLEVLEV(lspec,row,col)] );

      }

      fprintf( file_A, "\n" );
      fprintf( file_B, "\n" );
      fprintf( file_C, "\n" );

    }


    fclose(file_A);
    fclose(file_B);
    fclose(file_C);

  } /* end of lspec loop over line producing species */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_R: write the transition matrix R                                                        */
/*-----------------------------------------------------------------------------------------------*/

int write_R( std::string tag, long gridp, double *R )
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string lspec_name = species[ lspec_nr[lspec] ].sym;

    std::string file_name = OUTPUT_DIRECTORY + "R_" + lspec_name + tag + ".txt";

    FILE *file = fopen(file_name.c_str(), "w");


    if (file == NULL){

      printf("Error opening file!\n");
      std::cout << file_name + "\n";
      exit(1);
    }


    for (long row=0; row<nlev[lspec]; row++){

      for (long col=0; col<nlev[lspec]; col++){

        fprintf( file, "%lE\t", R[LSPECGRIDLEVLEV(lspec,gridp,row,col)] );

      }

      fprintf( file, "\n" );

    }

    fclose(file);

  } /* end of lspec loop over line producing species */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_transition_levels: write the levels corresponding to each transition                    */
/*-----------------------------------------------------------------------------------------------*/

int write_transition_levels( std::string tag, int *irad, int *jrad )
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string lspec_name = species[ lspec_nr[lspec] ].sym;

    std::string file_name = OUTPUT_DIRECTORY + "transition_levels_" + lspec_name + tag + ".txt";

    FILE *file = fopen(file_name.c_str(), "w");


    if (file == NULL){

      printf("Error opening file!\n");
      std::cout << file_name + "\n";
      exit(1);
    }


    for (int kr=0; kr<nrad[lspec]; kr++){

      int i = irad[LSPECRAD(lspec,kr)];          /* i level index corresponding to transition kr */
      int j = jrad[LSPECRAD(lspec,kr)];          /* j level index corresponding to transition kr */

      fprintf( file, "%d \t %d\n", i, j );

    }


    fclose(file);

  } /* end of lspec loop over line producing species */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_performance_log: write the performance results of the run                               */
/*-----------------------------------------------------------------------------------------------*/

int write_performance_log( double time_total, double time_level_pop, double time_chemistry,
                           double time_ray_tracing, int n_tb_iterations )
{


  std::string run_number = RUN_NUMBER;
  std::string file_name = "tests/performance/performance_log_" + run_number + ".txt";

  FILE *file = fopen(file_name.c_str(), "w");


  if (file == NULL){

    printf("Error opening file!\n");
    std::cout << file_name + "\n";
    exit(1);
  }

  fprintf( file, "NGRID            %d\n", NGRID );
  fprintf( file, "time_total       %lE\n", time_total );
  fprintf( file, "time_ray_tracing %lE\n", time_ray_tracing );
  fprintf( file, "time_level_pop   %lE\n", time_level_pop );
  fprintf( file, "time_chemistry   %lE\n", time_chemistry );
  fprintf( file, "n_tb_iterations  %d\n",  n_tb_iterations );


  fclose(file);




  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_LTE_deviation: write the relative deviation of the level populations from the LTE value */
/*-----------------------------------------------------------------------------------------------*/

int write_LTE_deviation( std::string tag, GRIDPOINT *gridpoint, double *energy, double* weight,
                         double *temperature_gas, double *pop )
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }


  double LTE_pop[NGRID*TOT_NLEV];                                        /* level population n_i */

  initialize_double_array(LTE_pop, NGRID*TOT_NLEV);

  calc_LTE_populations(gridpoint, energy, weight, temperature_gas, LTE_pop);


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string lspec_name = species[ lspec_nr[lspec] ].sym;

    std::string file_name = OUTPUT_DIRECTORY + "LTE_deviations_" + lspec_name + tag + ".txt";

    FILE *file = fopen(file_name.c_str(), "w");


    if (file == NULL){

      std :: cout << "Error opening file " << file_name << "!\n";
      std::cout << file_name + "\n";
      exit(1);
    }


    for (long n=0; n<NGRID; n++){

      for (int i=0; i<nlev[lspec]; i++){

        double dev = 2.0 * (pop[LSPECGRIDLEV(lspec,n, i)] - LTE_pop[LSPECGRIDLEV(lspec,n, i)])
                         / (pop[LSPECGRIDLEV(lspec,n, i)] + LTE_pop[LSPECGRIDLEV(lspec,n, i)]);

        fprintf(file, "%lE\t", dev);
      }

      fprintf(file, "\n");
    }


    fclose(file);

  } /* end of lspec loop over line producing species */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_true_level_populations: write the true level populations                                */
/*-----------------------------------------------------------------------------------------------*/

int write_true_level_populations( std::string tag, GRIDPOINT *gridpoint, double *pop )
{


  if ( !tag.empty() ){

    tag = "_" + tag;
  }


  for (int lspec=0; lspec<NLSPEC; lspec++){

    std::string lspec_name = species[ lspec_nr[lspec] ].sym;

    std::string file_name = OUTPUT_DIRECTORY + "true_level_populations_" + lspec_name + tag + ".txt";

    FILE *file = fopen(file_name.c_str(), "w");


    if (file == NULL){

      std :: cout << "Error opening file " << file_name << "!\n";
      std::cout << file_name + "\n";
      exit(1);
    }


    for (long n=0; n<NGRID; n++){

      for (int i=0; i<nlev[lspec]; i++){

        double rel = pop[LSPECGRIDLEV(lspec,n, i)]
                     / gridpoint[n].density / species[lspec_nr[lspec]].abn[n];

        fprintf(file, "%lE\t", rel);
      }

      fprintf(file, "\n");
    }


    fclose(file);

  } /* end of lspec loop over line producing species */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* write_vtu_output: write all physical variables to the vtu input grid                          */
/*-----------------------------------------------------------------------------------------------*/

int write_vtu_output( std::string grid_inputfile,
                      double *temperature_gas, double *temperature_dust )
{


  /* Read the data from the .vtu file */

  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader
    = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();

  reader->SetFileName(grid_inputfile.c_str());
  reader->Update();

  vtkUnstructuredGrid* ugrid = reader->GetOutput();


  /* Reformat the Magritte output to append it to the grid */

  vtkSmartPointer<vtkDoubleArray> temp_gas
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> temp_dust
    = vtkSmartPointer<vtkDoubleArray>::New();

  vtkSmartPointer<vtkDoubleArray> abn
    = vtkSmartPointer<vtkDoubleArray>::New();

  temp_gas->SetNumberOfComponents(1);
  temp_gas->SetNumberOfTuples(NGRID);
  temp_gas->SetName("temperature_gas");

  temp_dust->SetNumberOfComponents(1);
  temp_dust->SetNumberOfTuples(NGRID);
  temp_dust->SetName("temperature_dust");

  abn->SetNumberOfComponents(NSPEC);
  abn->SetNumberOfTuples(NGRID);
  abn->SetName("abundance");


  for (long n=0; n<NGRID; n++){

    temp_gas ->InsertValue(n, temperature_gas[n]);
    temp_dust->InsertValue(n, temperature_dust[n]);


    double abundance[NSPEC];

    for (int spec=0; spec<NSPEC; spec++){

      abundance[spec] = species[spec].abn[n];
    }

    abn->InsertTuple(n, abundance);

  } /* end of n loop over grid points */


  /* Add the new arrays to the grid */

  ugrid->GetCellData()->AddArray(temp_gas);
  ugrid->GetCellData()->AddArray(temp_dust);
  ugrid->GetCellData()->AddArray(abn);


  /* Write the .vtu file */

  std::string file_name = OUTPUT_DIRECTORY + "grid.vtu";

  vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer
    = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();

  writer->SetFileName(file_name.c_str());

# if VTK_MAJOR_VERSION <= 5
  writer->SetInput(ugrid);
# else
  writer->SetInputData(ugrid);
# endif
  writer->Write();


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/
