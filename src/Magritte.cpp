/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Magritte: main                                                                                */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
/*#include <mpi.h>*/

#include <string>
#include <iostream>

#include "declarations.hpp"
#include "definitions.hpp"

#include "initializers.hpp"
#include "species_tools.hpp"

#include "read_input.hpp"
#include "read_chemdata.hpp"
#include "read_linedata.hpp"

#include "create_healpixvectors.hpp"
#include "ray_tracing.hpp"

#include "calc_rad_surface.hpp"
#include "calc_column_density.hpp"
#include "calc_AV.hpp"
#include "calc_UV_field.hpp"
#include "calc_temperature_dust.hpp"
#include "chemistry.hpp"
#include "thermal_balance.hpp"
#include "update_temperature_gas.hpp"

#include "write_output.hpp"



/* main for Magritte                                                                             */
/*-----------------------------------------------------------------------------------------------*/

int main()
{


  /* Initialize all timers */

  double time_total = 0.0;                      /* total time in Magritte without writing output */

  time_total -= omp_get_wtime();


  double time_ray_tracing = 0.0;                                    /* total time in ray_tracing */
  double time_chemistry   = 0.0;                                     /* total time in abundances */
  double time_level_pop   = 0.0;                              /* total time in level_populations */




  printf("\n");
  printf("Magritte : \n");
  printf("\n");
  printf("Multidimensional Accelerated General-purpose RadIaTive TransEr \n");
  printf("\n");
  printf("-------------------------------------------------------------- \n");
  printf("\n");
  printf("\n");





  /*   READ GRID INPUT                                                                           */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): reading grid input \n");


  /* Define grid (using types defined in definitions.h)*/

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  /* NOTE: gridpoint does not have to be initialized as long as read_input works */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */

  initialize_evalpoint(evalpoint);


  /* Read input file */

  read_input(grid_inputfile, gridpoint);


  printf("(Magritte): grid input read \n\n");


  /*_____________________________________________________________________________________________*/





  /*   READ INPUT CHEMISTRY                                                                      */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): reading chemistry input \n");


  /* Read the species (and their initial abundances) */

  read_species(spec_datafile);


  /* Get and store the species numbers of some inportant species */

  e_nr    = get_species_nr("e-");                       /* species nr corresponding to electrons */

  H2_nr   = get_species_nr("H2");                              /* species nr corresponding to H2 */

  HD_nr   = get_species_nr("HD");                              /* species nr corresponding to HD */

  C_nr    = get_species_nr("C");                                /* species nr corresponding to C */

  H_nr    = get_species_nr("H");                                /* species nr corresponding to H */

  H2x_nr  = get_species_nr("H2+");                            /* species nr corresponding to H2+ */

  HCOx_nr = get_species_nr("HCO+");                          /* species nr corresponding to HCO+ */

  H3x_nr  = get_species_nr("H3+");                            /* species nr corresponding to H3+ */

  H3Ox_nr = get_species_nr("H3O+");                          /* species nr corresponding to H3O+ */

  Hex_nr  = get_species_nr("He+");                            /* species nr corresponding to He+ */

  CO_nr   = get_species_nr("CO");                              /* species nr corresponding to CO */


  /* Read the reactions */

  read_reactions(reac_datafile);


  printf("(Magritte): chemistry input read \n\n");


  /*_____________________________________________________________________________________________*/





  /*   DECLARE AND INITIALIZE LINE VARIABLES                                                     */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): declaring and initializing line variables \n");


  /* Setup the data structures which will store the line data */

  // setup_data_structures(line_datafile);


  /* Define line related variables */

  int irad[TOT_NRAD];           /* level index corresponding to radiative transition [0..nrad-1] */

  initialize_int_array(irad, TOT_NRAD);

  int jrad[TOT_NRAD];           /* level index corresponding to radiative transition [0..nrad-1] */

  initialize_int_array(jrad, TOT_NRAD);

  double energy[TOT_NLEV];                                                /* energy of the level */

  initialize_double_array(energy, TOT_NLEV);

  double weight[TOT_NLEV];                                    /* statistical weight of the level */

  initialize_double_array(weight, TOT_NLEV);

  double frequency[TOT_NLEV2];             /* photon frequency corresponing to i -> j transition */

  initialize_double_array(frequency, TOT_NLEV2);

  double A_coeff[TOT_NLEV2];                                        /* Einstein A_ij coefficient */

  initialize_double_array(A_coeff, TOT_NLEV2);

  double B_coeff[TOT_NLEV2];                                        /* Einstein B_ij coefficient */

  initialize_double_array(B_coeff, TOT_NLEV2);

  double C_coeff[TOT_NLEV2];                                        /* Einstein C_ij coefficient */

  initialize_double_array(C_coeff, TOT_NLEV2);

  double R[NGRID*TOT_NLEV2];                                           /* transition matrix R_ij */

  initialize_double_array(R, NGRID*TOT_NLEV2);


  /* Define the collision related variables */

  double coltemp[TOT_CUM_TOT_NCOLTEMP];               /* Collision temperatures for each partner */
                                                                   /*[NLSPEC][ncolpar][ncoltemp] */
  initialize_double_array(coltemp, TOT_CUM_TOT_NCOLTEMP);

  double C_data[TOT_CUM_TOT_NCOLTRANTEMP];           /* C_data for each partner, tran. and temp. */
                                                        /* [NLSPEC][ncolpar][ncoltran][ncoltemp] */
  initialize_double_array(C_data, TOT_CUM_TOT_NCOLTRANTEMP);

  int icol[TOT_CUM_TOT_NCOLTRAN];     /* level index corresp. to col. transition [0..ncoltran-1] */
                                                                  /* [NLSPEC][ncolpar][ncoltran] */
  initialize_int_array(icol, TOT_CUM_TOT_NCOLTRAN);

  int jcol[TOT_CUM_TOT_NCOLTRAN];     /* level index corresp. to col. transition [0..ncoltran-1] */
                                                                  /* [NLSPEC][ncolpar][ncoltran] */
  initialize_int_array(jcol, TOT_CUM_TOT_NCOLTRAN);


  /* Define the helper arrays specifying the species of the collisiopn partners */

  initialize_int_array(spec_par, TOT_NCOLPAR);

  initialize_char_array(ortho_para, TOT_NCOLPAR);


  printf("(Magritte): data structures are set up \n\n");


  /*_____________________________________________________________________________________________*/





  /*   READ LINE DATA FOR EACH LINE PRODUCING SPECIES                                            */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): reading line data \n");


  /* Read the line data files stored in the list(!) line_data */

  read_linedata( line_datafile, irad, jrad, energy, weight, frequency,
                 A_coeff, B_coeff, coltemp, C_data, icol, jcol );


  printf("(Magritte): line data read \n");


  /*_____________________________________________________________________________________________*/





  /*   CREATE HEALPIX VECTORS AND FIND ANTIPODAL PAIRS                                           */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): creating HEALPix vectors \n");


  /* Create the (unit) HEALPix vectors and find antipodal pairs */

  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  create_healpixvectors(unit_healpixvector, antipod);


  printf("(Magritte): HEALPix vectors created \n\n");


  /*_____________________________________________________________________________________________*/





  /*   RAY TRACING                                                                               */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): tracing rays \n");


  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NGRID*NGRID);

  initialize_long_array(raytot, NGRID*NRAYS);

  initialize_long_array(cum_raytot, NGRID*NRAYS);


  /* Execute ray_tracing */

  time_ray_tracing -= omp_get_wtime();

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);

  time_ray_tracing += omp_get_wtime();


  printf("\n(Magritte): time in ray_tracing: %lf sec \n", time_ray_tracing);


  printf("(Magritte): rays traced \n\n");


  /*_____________________________________________________________________________________________*/





  /*   CALCULATE EXTERNAL RADIATION FIELD                                                        */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): calculating external radiation field \n");


  double G_external[3];                                       /* external radiation field vector */

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  double rad_surface[NGRID*NRAYS];

  initialize_double_array(rad_surface, NGRID*NRAYS);


  /* Calculate the radiation surface */

  calc_rad_surface(G_external, unit_healpixvector, rad_surface);

  printf("(Magritte): external radiation field calculated \n");


  /*_____________________________________________________________________________________________*/





  /*   MAKE GUESS FOR GAS TEMPERATURE AND CALCULATE DUST TEMPERATURE                             */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): making a guess for gas temperature and calculating dust temperature \n");


  double temperature_dust[NGRID];                  /* temperature of the dust at each grid point */

  initialize_double_array(temperature_dust, NGRID);

  double column_tot[NGRID*NRAYS];                                        /* total column density */

  initialize_double_array(column_tot, NGRID*NRAYS);

  double AV[NGRID*NRAYS];                       /* Visual extinction (only takes into account H) */

  initialize_double_array(AV, NGRID*NRAYS);

  double UV_field[NGRID];

  initialize_double_array(UV_field, NGRID);


  /* Calculate the total column density */

  calc_column_density(gridpoint, evalpoint, column_tot, NSPEC-1);


  /* Calculate the visual extinction */

  calc_AV(column_tot, AV);


  /* Calculcate the UV field */

  calc_UV_field(antipod, AV, rad_surface, UV_field);


  double temperature_gas[NGRID];                           /* gas temperature at each grid point */

  guess_temperature_gas(UV_field, temperature_gas);
  // initialize_double_array_with_value(temperature_gas, 200.0, NGRID);


  /* Calculate the dust temperature */

  calc_temperature_dust(UV_field, rad_surface, temperature_dust);


  write_temperature_gas("guess", temperature_gas);

  write_temperature_dust("guess", temperature_dust);


  printf("(Magritte): gas temperature guessed and dust temperature calculated \n\n");


  /*_____________________________________________________________________________________________*/





  /*   PRELIMINARY CHEMISTRY ITERATIONS                                                          */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): starting preliminary chemistry iterations \n\n");


  double column_H[NGRID*NRAYS];                  /* H column density for each ray and grid point */

  initialize_double_array(column_H, NGRID*NRAYS);

  double column_H2[NGRID*NRAYS];                /* H2 column density for each ray and grid point */

  initialize_double_array(column_H2, NGRID*NRAYS);

  double column_HD[NGRID*NRAYS];                /* HD column density for each ray and grid point */

  initialize_double_array(column_HD, NGRID*NRAYS);

  double column_C[NGRID*NRAYS];                  /* C column density for each ray and grid point */

  initialize_double_array(column_C, NGRID*NRAYS);

  double column_CO[NGRID*NRAYS];                /* CO column density for each ray and grid point */

  initialize_double_array(column_CO, NGRID*NRAYS);


  /* Preliminary chemistry iterations */

  int np_chem_iterations = 5;                /* total number of preliminary chemistry iterations */


  for (int chem_iteration=0; chem_iteration<np_chem_iterations; chem_iteration++){

    printf("(Magritte):   chemistry iteration %d of %d \n", chem_iteration+1, np_chem_iterations);


    /* Calculate column densities */

    calc_column_density(gridpoint, evalpoint, column_H, H_nr);
    calc_column_density(gridpoint, evalpoint, column_H2, H2_nr);
    calc_column_density(gridpoint, evalpoint, column_HD, HD_nr);
    calc_column_density(gridpoint, evalpoint, column_C, C_nr);
    calc_column_density(gridpoint, evalpoint, column_CO, CO_nr);


    /* Calculate the chemical abundances given the current temperatures and radiation field */

    time_chemistry -= omp_get_wtime();

    chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
               column_H2, column_HD, column_C, column_CO );

    time_chemistry += omp_get_wtime();

  } /* End of chemistry iteration */


  printf("\n(Magritte): preliminary chemistry iterations done \n\n");


  /*_____________________________________________________________________________________________*/





  /*   PRELIMINARY THERMAL BALANCE ITERATIONS                                                    */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): calculating the minimal and maximal thermal flux \n\n");


  double mean_intensity[NGRID*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NGRID*TOT_NRAD);

  double pop[NGRID*TOT_NLEV];                                            /* level population n_i */

  initialize_double_array(pop, NGRID*TOT_NLEV);

  double prev1_pop[NGRID*TOT_NLEV];                      /* level population n_i 1 iteration ago */

  initialize_double_array(prev1_pop, NGRID*TOT_NLEV);

  double prev2_pop[NGRID*TOT_NLEV];                     /* level population n_i 2 iterations ago */

  initialize_double_array(prev2_pop, NGRID*TOT_NLEV);

  double prev3_pop[NGRID*TOT_NLEV];                     /* level population n_i 3 iterations ago */

  initialize_double_array(prev3_pop, NGRID*TOT_NLEV);


  double temperature_a[NGRID];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_a, NGRID);

  double temperature_b[NGRID];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_b, NGRID);

  double temperature_c[NGRID];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_c, NGRID);

  double temperature_d[NGRID];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_d, NGRID);

  double temperature_e[NGRID];                                 /* variable for Brent's algorithm */

  initialize_double_array(temperature_e, NGRID);

  double thermal_ratio_a[NGRID];                               /* variable for Brent's algorithm */

  initialize_double_array(thermal_ratio_a, NGRID);

  double thermal_ratio_b[NGRID];                               /* variable for Brent's algorithm */

  initialize_double_array(thermal_ratio_b, NGRID);

  double thermal_ratio_c[NGRID];                               /* variable for Brent's algorithm */

  initialize_double_array(thermal_ratio_c, NGRID);


  double thermal_ratio[NGRID];

  initialize_double_array(thermal_ratio, NGRID);


  double thermal_sum[NGRID];

  initialize_double_array(thermal_sum, NGRID);


  double prev2_temperature_gas[NGRID];

  initialize_previous_temperature_gas(prev2_temperature_gas, temperature_gas);


  for (int iter=0; iter<10; iter++){


    thermal_balance_iteration( gridpoint, evalpoint, antipod, column_H, column_H2, column_HD,
                               column_C, column_CO, UV_field, temperature_gas, temperature_dust,
                               rad_surface, AV, irad, jrad, energy, weight, frequency,
                               A_coeff, B_coeff, C_coeff, R, C_data, coltemp, icol, jcol,
                               prev1_pop, prev2_pop, prev3_pop, pop, mean_intensity, thermal_ratio,
                               thermal_sum,
                               time_chemistry, time_level_pop );


    initialize_double_array_with(thermal_ratio_b, thermal_ratio, NGRID);


    for (long gridp=0; gridp<NGRID; gridp++){

      update_temperature_gas( thermal_ratio, thermal_sum, gridp, temperature_gas, prev2_temperature_gas,
                              temperature_a, temperature_b, thermal_ratio_a, thermal_ratio_b );

    }

  }

  initialize_double_array_with(temperature_gas, temperature_b, NGRID);

  write_double_1("temperature_a", "", NGRID, temperature_a );
  write_double_1("temperature_b", "", NGRID, temperature_b );


  // /* Thermal ratio for minimum temperature */
  //
  // initialize_double_array_with_value(temperature_gas, T_CMB, NGRID);
  //
  // thermal_balance_iteration( gridpoint, evalpoint, antipod, column_H, column_H2, column_HD,
  //                            column_C, column_CO, UV_field, temperature_gas, temperature_dust,
  //                            rad_surface, AV, irad, jrad, energy, weight, frequency,
  //                            A_coeff, B_coeff, C_coeff, R, C_data, coltemp, icol, jcol,
  //                            prev1_pop, prev2_pop, prev3_pop, pop, mean_intensity, thermal_ratio,
  //                            time_chemistry, time_level_pop );
  //
  // initialize_double_array_with(thermal_ratio_a, thermal_ratio, NGRID);
  //
  //
  // /* Thermal ratio for maximum temperature */
  //
  // initialize_double_array_with_value(temperature_gas, 100.0, NGRID);
  //
  // thermal_balance_iteration( gridpoint, evalpoint, antipod, column_H, column_H2, column_HD,
  //                            column_C, column_CO, UV_field, temperature_gas, temperature_dust,
  //                            rad_surface, AV, irad, jrad, energy, weight, frequency,
  //                            A_coeff, B_coeff, C_coeff, R, C_data, coltemp, icol, jcol,
  //                            prev1_pop, prev2_pop, prev3_pop, pop, mean_intensity, thermal_ratio,
  //                            time_chemistry, time_level_pop );
  //
  // initialize_double_array_with(thermal_ratio_a, thermal_ratio, NGRID);
  //
  //
  // for (long gridp=0; gridp<NGRID; gridp++){
  //
  //   shuffle_Brent( gridp, temperature_a, temperature_b, temperature_c, temperature_d,
  //                  temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c );
  //
  //
  //   update_temperature_gas_Brent( gridp, temperature_a, temperature_b, temperature_c,
  //                                 temperature_d, temperature_e, thermal_ratio_a,
  //                                 thermal_ratio_b, thermal_ratio_c );
  //
  //   if(temperature_b[gridp] < T_CMB){
  //
  //     temperature_b[gridp] = T_CMB;
  //   }
  //
  //   else if (temperature_b[gridp] > 30000.0){
  //
  //     temperature_b[gridp] = 30000.0;
  //   }
  //
  //   temperature_gas[gridp] = temperature_b[gridp];
  //
  //   printf("temp is %lf\n", temperature_gas[gridp]);
  //
  //
  // } /* end of gridp loop over grid points */


  printf("(Magritte): minimal and maximal thermal flux calculated \n\n");


  /*_____________________________________________________________________________________________*/





  /*   CALCULATE THERMAL BALANCE (ITERATIVELY)                                                   */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): starting thermal balance iterations \n\n");


  bool somewhere_no_thermal_balance = true;

  bool no_thermal_balance[NGRID];

  initialize_bool(true, NGRID, no_thermal_balance);

  int niterations = 0;



  /* Thermal balance iterations */



  while (somewhere_no_thermal_balance){

    somewhere_no_thermal_balance = false;

    niterations++;


    printf("(Magritte): thermal balance iteration %d\n", niterations);


    long n_not_converged = 0;                /* number of grid points that are not yet converged */


    thermal_balance_iteration( gridpoint, evalpoint, antipod, column_H, column_H2, column_HD,
                               column_C, column_CO, UV_field, temperature_gas, temperature_dust,
                               rad_surface, AV, irad, jrad, energy, weight, frequency,
                               A_coeff, B_coeff, C_coeff, R, C_data, coltemp, icol, jcol,
                               prev1_pop, prev2_pop, prev3_pop, pop, mean_intensity, thermal_ratio,
                               thermal_sum,
                               time_chemistry, time_level_pop );


    // initialize_double_array_with(thermal_ratio_b, thermal_ratio, NGRID);

    for (long n=0; n<NGRID; n++){

      thermal_ratio_b[n] = thermal_ratio[n] * thermal_sum[n] * thermal_sum[n];
    }


    /* Calculate the thermal balance for each gridpoint */

    for (long gridp=0; gridp<NGRID; gridp++){

      shuffle_Brent( gridp, temperature_a, temperature_b, temperature_c, temperature_d,
                     temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c );


      /* Check for thermal balance (convergence) */

      if (fabs(thermal_ratio[gridp]) > THERMAL_PREC){

        update_temperature_gas_Brent( gridp, temperature_a, temperature_b, temperature_c,
                                      temperature_d, temperature_e, thermal_ratio_a,
                                      thermal_ratio_b, thermal_ratio_c );

        if(temperature_b[gridp] < T_CMB){

          temperature_b[gridp] = T_CMB;
        }

        else if (temperature_b[gridp] > 30000.0){

          temperature_b[gridp] = 30000.0;
        }

        temperature_gas[gridp] = temperature_b[gridp];


        temperature_gas[gridp] = temperature_b[gridp];



        // update_temperature_gas(thermal_ratio, gridp, temperature_gas, prev2_temperature_gas );



        if (temperature_gas[gridp] != T_CMB){

          no_thermal_balance[gridp] = true;

          somewhere_no_thermal_balance = true;

          n_not_converged++;
        }

      }


    } /* end of gridp loop over grid points */


    printf("(Magritte): heating and cooling calculated \n\n");


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Limit the number of iterations */

    if (niterations > MAX_NITERATIONS || n_not_converged < NGRID/20){

      somewhere_no_thermal_balance = false;
    }


    printf("(Magritte): Not yet converged for %ld of %d\n", n_not_converged, NGRID);


  } /* end of thermal balance iterations */


  printf("(Magritte): thermal balance reached in %d iterations \n\n", niterations);


  /*_____________________________________________________________________________________________*/





  time_total += omp_get_wtime();





  /*   WRITE OUTPUT                                                                              */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): writing output \n");


  /* Write the output files */

  std::string tag = "final";

  write_abundances(tag);

  write_level_populations(tag, pop);

  write_line_intensities(tag, mean_intensity);

  write_temperature_gas(tag, temperature_gas);

  write_temperature_dust(tag, temperature_dust);

  write_performance_log(time_total, time_level_pop, time_chemistry, time_ray_tracing, niterations);


  printf("(Magritte): output written \n\n");


  /*_____________________________________________________________________________________________*/




  printf("(Magritte): performance of this run : \n\n");




  printf("(Magritte): done \n\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
