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


# ifndef ON_THE_FLY

  double R[NGRID*TOT_NLEV2];                                           /* transition matrix R_ij */

  initialize_double_array(R, NGRID*TOT_NLEV2);

# endif


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



# ifndef ON_THE_FLY


  /*   RAY TRACING                                                                               */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): tracing rays (not ON_THE_FLY) \n");


  /* Declare and initialize the evaluation points */

  long key[NGRID*NGRID];              /* stores the nrs. of the grid points on the rays in order */

  long raytot[NGRID*NRAYS];                /* cumulative nr. of evaluation points along each ray */

  long cum_raytot[NGRID*NRAYS];            /* cumulative nr. of evaluation points along each ray */


  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */


  /* Execute ray_tracing */

  time_ray_tracing -= omp_get_wtime();

  ray_tracing(gridpoint, evalpoint, key, raytot, cum_raytot);

  time_ray_tracing += omp_get_wtime();


  printf("\n(Magritte): time in ray_tracing: %lf sec \n", time_ray_tracing);


  printf("(Magritte): rays traced \n\n");


  /*_____________________________________________________________________________________________*/


# endif



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

  calc_rad_surface(G_external, rad_surface);

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


# ifdef ON_THE_FLY

  calc_column_density(gridpoint, column_tot, NSPEC-1);

# else

  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_tot, NSPEC-1);

# endif


  /* Calculate the visual extinction */

  calc_AV(column_tot, AV);


  /* Calculcate the UV field */

  calc_UV_field(AV, rad_surface, UV_field);


  double temperature_gas[NGRID];                           /* gas temperature at each grid point */

  guess_temperature_gas(UV_field, temperature_gas);


  /* Calculate the dust temperature */

  calc_temperature_dust(UV_field, rad_surface, temperature_dust);


  write_double_2("column_tot", "", NGRID, NRAYS, column_tot);


  // write_temperature_gas("guess", temperature_gas);

  // write_temperature_dust("guess", temperature_dust);


  printf("(Magritte): gas temperature guessed and dust temperature calculated \n\n");


  /*_____________________________________________________________________________________________*/





  /*   PRELIMINARY CHEMISTRY ITERATIONS                                                          */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): starting preliminary chemistry iterations \n\n");


  double column_H2[NGRID*NRAYS];                /* H2 column density for each ray and grid point */

  initialize_double_array(column_H2, NGRID*NRAYS);

  double column_HD[NGRID*NRAYS];                /* HD column density for each ray and grid point */

  initialize_double_array(column_HD, NGRID*NRAYS);

  double column_C[NGRID*NRAYS];                  /* C column density for each ray and grid point */

  initialize_double_array(column_C, NGRID*NRAYS);

  double column_CO[NGRID*NRAYS];                /* CO column density for each ray and grid point */

  initialize_double_array(column_CO, NGRID*NRAYS);


  /* Preliminary chemistry iterations */

  for (int chem_iteration=0; chem_iteration<PRELIM_CHEM_ITER; chem_iteration++){

    printf("(Magritte):   chemistry iteration %d of %d \n", chem_iteration+1, PRELIM_CHEM_ITER);


    /* Calculate the chemical abundances given the current temperatures and radiation field */

    time_chemistry -= omp_get_wtime();


#   ifdef ON_THE_FLY

    chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
               column_H2, column_HD, column_C, column_CO );

#   else

    chemistry( gridpoint, evalpoint, key, raytot, cum_raytot, temperature_gas, temperature_dust,
               rad_surface, AV, column_H2, column_HD, column_C, column_CO );

#   endif


    time_chemistry += omp_get_wtime();

  } /* End of chemistry iteration */


  printf("\n(Magritte): preliminary chemistry iterations done \n\n");


  /*_____________________________________________________________________________________________*/





  /*   PRELIMINARY THERMAL BALANCE ITERATIONS                                                    */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): calculating the minimal and maximal thermal flux \n\n");


  double mean_intensity[NGRID*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NGRID*TOT_NRAD);


  double mean_intensity_eff[NGRID*TOT_NRAD];                         /* mean intensity for a ray */

  initialize_double_array(mean_intensity_eff, NGRID*TOT_NRAD);

  double Lambda_diagonal[NGRID*TOT_NRAD];                            /* mean intensity for a ray */

  initialize_double_array(Lambda_diagonal, NGRID*TOT_NRAD);


  double pop[NGRID*TOT_NLEV];                                            /* level population n_i */

  initialize_double_array(pop, NGRID*TOT_NLEV);


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


  double prev_temperature_gas[NGRID];

  initialize_previous_temperature_gas(prev_temperature_gas, temperature_gas);


  for (int tb_iteration=0; tb_iteration<PRELIM_TB_ITER; tb_iteration++){

    printf("(Magritte):   thermal balance iteration %d of %d \n", tb_iteration+1, PRELIM_TB_ITER);


#   ifdef ON_THE_FLY

    thermal_balance_iteration( gridpoint, column_H2, column_HD, column_C, column_CO,
                               UV_field, temperature_gas, temperature_dust,
                               rad_surface, AV, irad, jrad, energy, weight, frequency,
                               A_coeff, B_coeff, C_data, coltemp, icol, jcol,
                               pop, mean_intensity,
                               Lambda_diagonal, mean_intensity_eff,
                               thermal_ratio,
                               &time_chemistry, &time_level_pop );

#   else

    thermal_balance_iteration( gridpoint, evalpoint, key, raytot, cum_raytot,
                               column_H2, column_HD, column_C,
                               column_CO, UV_field, temperature_gas, temperature_dust,
                               rad_surface, AV, irad, jrad, energy, weight, frequency,
                               A_coeff, B_coeff, R, C_data, coltemp, icol, jcol,
                               pop, mean_intensity,
                               Lambda_diagonal, mean_intensity_eff,
                               thermal_ratio,
                               &time_chemistry, &time_level_pop );

#   endif


    initialize_double_array_with(thermal_ratio_b, thermal_ratio, NGRID);


    for (long gridp=0; gridp<NGRID; gridp++){

      update_temperature_gas( thermal_ratio, gridp, temperature_gas, prev_temperature_gas,
                              temperature_a, temperature_b, thermal_ratio_a, thermal_ratio_b );

    }

  }

  initialize_double_array_with(temperature_gas, temperature_b, NGRID);

  write_double_1("temperature_a", "", NGRID, temperature_a );
  write_double_1("temperature_b", "", NGRID, temperature_b );


  printf("(Magritte): minimal and maximal thermal flux calculated \n\n");


  /*_____________________________________________________________________________________________*/





  /*   CALCULATE THERMAL BALANCE (ITERATIVELY)                                                   */
  /*_____________________________________________________________________________________________*/


  printf("(Magritte): starting thermal balance iterations \n\n");


  bool no_thermal_balance = true;  /* true when the temperature of a grid point is not converged */

  int niterations = 0;


  /* Thermal balance iterations */

  while (no_thermal_balance){

    no_thermal_balance = false;

    niterations++;


    printf("(Magritte): thermal balance iteration %d\n", niterations);


    long n_not_converged = 0;                /* number of grid points that are not yet converged */


#   ifdef ON_THE_FLY

    thermal_balance_iteration( gridpoint, column_H2, column_HD, column_C, column_CO,
                               UV_field, temperature_gas, temperature_dust,
                               rad_surface, AV, irad, jrad, energy, weight, frequency,
                               A_coeff, B_coeff, C_data, coltemp, icol, jcol,
                               pop, mean_intensity,
                               Lambda_diagonal, mean_intensity_eff,
                               thermal_ratio,
                               &time_chemistry, &time_level_pop );

#   else

    thermal_balance_iteration( gridpoint, evalpoint, key, raytot, cum_raytot,
                               column_H2, column_HD, column_C,
                               column_CO, UV_field, temperature_gas, temperature_dust,
                               rad_surface, AV, irad, jrad, energy, weight, frequency,
                               A_coeff, B_coeff, R, C_data, coltemp, icol, jcol,
                               pop, mean_intensity,
                               Lambda_diagonal, mean_intensity_eff,
                               thermal_ratio,
                               &time_chemistry, &time_level_pop );

#   endif


    initialize_double_array_with(thermal_ratio_b, thermal_ratio, NGRID);


    /* Calculate the thermal balance for each gridpoint */

    for (long gridp=0; gridp<NGRID; gridp++){

      shuffle_Brent( gridp, temperature_a, temperature_b, temperature_c, temperature_d,
                     temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c );


      /* Check for thermal balance (convergence) */

      if (fabs(thermal_ratio[gridp]) > THERMAL_PREC){

        update_temperature_gas_Brent( gridp, temperature_a, temperature_b, temperature_c,
                                      temperature_d, temperature_e, thermal_ratio_a,
                                      thermal_ratio_b, thermal_ratio_c );


        temperature_gas[gridp] = temperature_b[gridp];


        if (temperature_gas[gridp] != T_CMB){

          no_thermal_balance = true;

          n_not_converged++;
        }

      }


    } /* end of gridp loop over grid points */


    printf("(Magritte): heating and cooling calculated \n\n");


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Limit the number of iterations */

    if (niterations > MAX_NITERATIONS || n_not_converged < NGRID/20){

      no_thermal_balance = false;
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





  printf("(Magritte): done \n\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
