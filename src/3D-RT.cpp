/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* 3D-RT: main                                                                                   */
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
using namespace std;

#include "declarations.hpp"
#include "definitions.hpp"

#include "initializers.hpp"
#include "species_tools.hpp"
#include "data_tools.hpp"
#include "setup_data_structures.hpp"

#include "read_input.hpp"
#include "read_chemdata.hpp"
#include "read_linedata.hpp"

#include "create_healpixvectors.hpp"
#include "ray_tracing.hpp"

#include "rad_surface_calculator.hpp"
#include "column_density_calculator.hpp"
#include "AV_calculator.hpp"
#include "UV_field_calculator.hpp"
#include "dust_temperature_calculation.hpp"
#include "abundances.hpp"
#include "level_populations.hpp"
#include "heating.hpp"
#include "cooling.hpp"
#include "update_temperature_gas.hpp"

#include "write_output.hpp"



/* main code for 3D-RT: 3D Radiative Transfer                                                    */
/*-----------------------------------------------------------------------------------------------*/

int main()
{

  double time_ray_tracing = 0.0;                                    /* total time in ray_tracing */
  double time_abundances = 0.0;                                      /* total time in abundances */
  double time_level_pop = 0.0;                                /* total time in level_populations */


  /* Temporary */

  metallicity = 1.0;

  gas_to_dust = 100.0;

  double v_turb = 1.0;





  printf("                              \n");
  printf("3D-RT : 3D Radiative Transfer \n");
  printf("----------------------------- \n");
  printf("                              \n");





  /*   READ GRID INPUT                                                                           */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading grid input \n");


  /* Define grid (using types defined in definitions.h)*/

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  /* NOTE: gridpoint does not have to be initialized as long as read_input works */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */

  initialize_evalpoint(evalpoint);


  /* Read input file */

  read_input(grid_inputfile, gridpoint);


  printf("(3D-RT): grid input read \n\n");


  /*_____________________________________________________________________________________________*/





  /*   READ INPUT CHEMISTRY                                                                      */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading chemistry input \n");


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


  printf("(3D-RT): chemistry input read \n\n");


  /*_____________________________________________________________________________________________*/





  /*   SETUP DATA STRUCTURES                                                                     */
  /*_____________________________________________________________________________________________*/


  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NGRID*NGRID);

  initialize_long_array(raytot, NGRID*NRAYS);

  initialize_long_array(cum_raytot, NGRID*NRAYS);


  /* Setup the data structures which will store the line data */

  setup_data_structures(line_datafile);


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


  /*_____________________________________________________________________________________________*/



//

  /*   READ LINE DATA FOR EACH LINE PRODUCING SPECIES                                            */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading line data \n");


  /* Read the line data files stored in the list(!) line_data */

  read_linedata( line_datafile, irad, jrad, energy, weight, frequency,
                 A_coeff, B_coeff, coltemp, C_data, icol, jcol );


  printf("(3D-RT): line data read \n");


  /*_____________________________________________________________________________________________*/





  /*   CREATE HEALPIX VECTORS AND FIND ANTIPODAL PAIRS                                           */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): creating HEALPix vectors \n");


  /* Create the (unit) HEALPix vectors and find antipodal pairs */

  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  create_healpixvectors(unit_healpixvector, antipod);


  printf("(3D-RT): HEALPix vectors creatied \n\n");


  /*_____________________________________________________________________________________________*/





  /*   RAY TRACING                                                                               */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): tracing rays \n");


  /* Execute ray_tracing */

  time_ray_tracing -= omp_get_wtime();

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);

  time_ray_tracing += omp_get_wtime();


  printf("\n(3D-RT): time in ray_tracing: %lf sec \n", time_ray_tracing);


  printf("(3D-RT): rays traced \n\n");


  /*_____________________________________________________________________________________________*/





  /*   CALCULATE THE EXTERNAL RADIATION FIELD                                                    */
  /*_____________________________________________________________________________________________*/


  double G_external[3];                                       /* external radiation field vector */

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  double rad_surface[NGRID*NRAYS];

  initialize_double_array(rad_surface, NGRID*NRAYS);


  /* Calculate the radiation surface */

  rad_surface_calculator(G_external, unit_healpixvector, rad_surface);


  /*_____________________________________________________________________________________________*/





  /*   CALCULATE THERMAL BALANCE (ITERATIVELY)                                                   */
  /*_____________________________________________________________________________________________*/


  double temperature_gas[NGRID];                    /* temperature of the gas at each grid point */

  initialize_temperature_gas(temperature_gas);

  double previous_temperature_gas[NGRID];    /* temp. of gas at each grid point, prev. iteration */

  initialize_previous_temperature_gas(previous_temperature_gas, temperature_gas);

  double temperature_dust[NGRID];                  /* temperature of the dust at each grid point */

  initialize_double_array(temperature_dust, NGRID);

  double column_H2[NGRID*NRAYS];                /* H2 column density for each ray and grid point */

  initialize_double_array(column_H2, NGRID*NRAYS);

  double column_HD[NGRID*NRAYS];                /* HD column density for each ray and grid point */

  initialize_double_array(column_HD, NGRID*NRAYS);

  double column_C[NGRID*NRAYS];                  /* C column density for each ray and grid point */

  initialize_double_array(column_C, NGRID*NRAYS);

  double column_CO[NGRID*NRAYS];                /* CO column density for each ray and grid point */

  initialize_double_array(column_CO, NGRID*NRAYS);

  double AV[NGRID*NRAYS];                       /* Visual extinction (only takes into account H) */

  initialize_double_array(AV, NGRID*NRAYS);

  double UV_field[NGRID];

  initialize_double_array(UV_field, NGRID);

  double pop[NGRID*TOT_NLEV];                                            /* level population n_i */

  initialize_level_populations(energy, temperature_gas, pop);

  double dpop[NGRID*TOT_NLEV];        /* change in level population n_i w.r.t previous iteration */

  initialize_double_array(dpop, NGRID*TOT_NLEV);

  double mean_intensity[NGRID*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NGRID*TOT_NRAD);

  bool no_thermal_balance = true;

  int niterations = 0;                                                   /* number of iterations */



  /* Thermal balance iterations */

  printf("(3D-RT): Starting thermal balance iterations \n\n");

  while (no_thermal_balance){

    no_thermal_balance = false;

    niterations++;


    printf("(3D-RT): thermal balance iteration %d\n", niterations);


    /* Calculate column densities */

    column_density_calculator(gridpoint, evalpoint, column_H2, H2_nr);
    column_density_calculator(gridpoint, evalpoint, column_HD, HD_nr);
    column_density_calculator(gridpoint, evalpoint, column_C, C_nr);
    column_density_calculator(gridpoint, evalpoint, column_CO, CO_nr);


    /* Calculate the visual extinction */

    AV_calculator(column_H2, AV);


    /* Calculcate the UV field */

    UV_field_calculator(AV, rad_surface, UV_field);


    /* Calculate the dust temperature */

    dust_temperature_calculation(UV_field, rad_surface, temperature_dust);


    /*   CALCULATE CHEMICAL ABUNDANCES                                                           */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    printf("(3D-RT): calculating chemical abundances \n\n");


    /* Calculate the chemical abundances by solving the rate equations */

    time_abundances -= omp_get_wtime();

    abundances( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
                column_H2, column_HD, column_C, column_CO, v_turb );

    time_abundances += omp_get_wtime();


    printf("\n(3D-RT): time in abundances: %lf sec\n", time_abundances);


    printf("(3D-RT): chemical abundances calculated \n\n");


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   CALCULATE LEVEL POPULATIONS (ITERATIVELY)                                               */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    printf("(3D-RT): calculating level populations \n\n");


    /* Calculate level populations for each line producing species */

    time_level_pop -= omp_get_wtime();

    level_populations( gridpoint, evalpoint, antipod, irad, jrad, frequency, v_turb,
                       A_coeff, B_coeff, C_coeff, R, pop, dpop, C_data,
                       coltemp, icol, jcol, temperature_gas, temperature_dust,
                       weight, energy, mean_intensity );

    time_level_pop += omp_get_wtime();


    printf("\n(3D-RT): time in level_populations: %lf sec\n", time_level_pop);


    printf("(3D-RT): level populations calculated \n\n");


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   CALCULATE HEATING AND COOLING                                                           */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    printf("(3D-RT): calculating heating and cooling \n\n");


    /* Calculate the thermal balance for each gridpoint */

    for (long gridp=0; gridp<NGRID; gridp++){

      double heating_total = heating( gridpoint, gridp, temperature_gas, temperature_dust,
                                      UV_field, v_turb );

      double cooling_total = cooling( gridp, irad, jrad, A_coeff, B_coeff, frequency,
                                      pop, mean_intensity );


      double thermal_flux = heating_total - cooling_total;

      double thermal_ratio = 0.0;

      if( fabs(heating_total + cooling_total) > 0.0 ){

        thermal_ratio = 2.0 * fabs(thermal_flux) / fabs(heating_total + cooling_total);
      }

      cout << "Thermal flux  is = " << thermal_flux << "\n";
      cout << "Thermal ratio is = " << thermal_ratio << "\n";

      cout << "min  = " << fabs(thermal_flux) << "\n";
      cout << "plus = " << fabs(heating_total + cooling_total) << "\n";

      cout << "Heating " << heating_total << "\n";
      cout << "Coolimg " << cooling_total << "\n";

      /* Check for thermal balance (convergence) */

      if (thermal_ratio > THERMAL_PREC){

        no_thermal_balance = true;

        update_temperature_gas(thermal_flux, gridp, temperature_gas, previous_temperature_gas );

      }


      cout << "gas temperature " << temperature_gas[gridp] << "\n";

    } /* end of gridp loop over grid points */

    printf("(3D-RT): heating and cooling calculated \n\n");


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Limit the number of iterations */

    if (niterations >= MAX_NITERATIONS){

      no_thermal_balance = false;
    }


  } /* end of thermal balance iterations */


  printf("(3D-RT): thermal balance reached in %d iterations \n\n", niterations);


  /*_____________________________________________________________________________________________*/





  /*   WRITE OUTPUT                                                                              */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): writing output \n");


  /* Write the output files  */

  string tag = "";

  write_abundances(tag);

  write_level_populations(tag, line_datafile, pop);

  write_line_intensities(tag, line_datafile, mean_intensity);

  write_temperature_gas(tag, temperature_gas);

  write_temperature_dust(tag, temperature_dust);


  printf("(3D-RT): output written \n\n");


  /*_____________________________________________________________________________________________*/





  printf("(3D-RT): done \n\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
