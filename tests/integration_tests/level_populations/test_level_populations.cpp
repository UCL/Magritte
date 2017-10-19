/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_level_populations:                                                                       */
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

#include <string>
#include <iostream>
using namespace std;

#include "catch.hpp"

#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"

#include "../../../src/initializers.hpp"
#include "../../../src/species_tools.hpp"
#include "../../../src/data_tools.hpp"
#include "../../../src/setup_data_structures.hpp"

#include "../../../src/read_input.hpp"
#include "../../../src/read_chemdata.hpp"
#include "../../../src/read_linedata.hpp"

#include "../../../src/create_healpixvectors.hpp"
#include "../../../src/ray_tracing.hpp"

#include "../../../src/calc_rad_surface.hpp"
#include "../../../src/calc_column_density.hpp"
#include "../../../src/calc_AV.hpp"
#include "../../../src/calc_UV_field.hpp"
#include "../../../src/calc_temperature_dust.hpp"
#include "../../../src/chemistry.hpp"
#include "../../../src/calc_LTE_populations.hpp"
#include "../../../src/level_populations.hpp"
#include "../../../src/heating.hpp"
#include "../../../src/cooling.hpp"
#include "../../../src/update_temperature_gas.hpp"

#include "../../../src/write_output.hpp"





TEST_CASE("Test level populations"){


  metallicity = 1.0;

  gas_to_dust = 100.0;

  double v_turb = 1.0;

  v_turb = 1.0E5 * v_turb;


  /* Since the executables are now in the directory /tests, we have to change the paths */

  string test_grid_inputfile = "../../../" + grid_inputfile;

  string test_spec_datafile  = "../../../" + spec_datafile;

  string test_reac_datafile  = "../../../" + reac_datafile;

  string test_line_datafile[NLSPEC];


  for(int l=0; l<NLSPEC; l++){

    test_line_datafile[l] = "../../../" + line_datafile[l];
  }


  /* Define grid (using types defined in definitions.h)*/

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  /* NOTE: gridpoint does not have to be initialized as long as read_input works */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */

  initialize_evalpoint(evalpoint);


  /* Read input file */

  cout << "   ! file :" << grid_inputfile << "\n";

  read_input(test_grid_inputfile, gridpoint);


  /* Read the species (and their initial abundances) */

  read_species(test_spec_datafile);


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

  read_reactions(test_reac_datafile);


  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NGRID*NGRID);

  initialize_long_array(raytot, NGRID*NRAYS);

  initialize_long_array(cum_raytot, NGRID*NRAYS);


  /* Setup the data structures which will store the line data */

  setup_data_structures(test_line_datafile);


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


  /* Read the line data files stored in the list(!) line_data */

  read_linedata( test_line_datafile, irad, jrad, energy, weight, frequency,
                 A_coeff, B_coeff, coltemp, C_data, icol, jcol );


  /* Create the (unit) HEALPix vectors and find antipodal pairs */

  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  create_healpixvectors(unit_healpixvector, antipod);


  /* Execute ray_tracing */

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);


  double G_external[3];                                       /* external radiation field vector */

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  double rad_surface[NGRID*NRAYS];

  initialize_double_array(rad_surface, NGRID*NRAYS);


  /* Calculate the radiation surface */

  calc_rad_surface(G_external, unit_healpixvector, rad_surface);



  double temperature_dust[NGRID];                  /* temperature of the dust at each grid point */

  initialize_double_array(temperature_dust, NGRID);

  double column_tot[NGRID*NRAYS];

  initialize_double_array(column_tot, NGRID*NRAYS);

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

  double AV[NGRID*NRAYS];                       /* Visual extinction (only takes into account H) */

  initialize_double_array(AV, NGRID*NRAYS);

  double UV_field[NGRID];

  initialize_double_array(UV_field, NGRID);

  double dpop[NGRID*TOT_NLEV];        /* change in level population n_i w.r.t previous iteration */

  initialize_double_array(dpop, NGRID*TOT_NLEV);

  double mean_intensity[NGRID*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NGRID*TOT_NRAD);


  /* Make a guess for the gas temperature, based on he UV field */

  calc_column_density(gridpoint, evalpoint, column_tot, NSPEC-1);

  calc_AV(column_tot, AV);

  calc_UV_field(antipod, AV, rad_surface, UV_field);

  double temperature_gas[NGRID];                    /* temperature of the gas at each grid point */

  guess_temperature_gas(UV_field, temperature_gas);

  double previous_temperature_gas[NGRID];    /* temp. of gas at each grid point, prev. iteration */

  initialize_previous_temperature_gas(previous_temperature_gas, temperature_gas);


  /* Preliminary chemistry iterations */

  for (int chem_iteration=0; chem_iteration<8; chem_iteration++){


    /* Calculate column densities */

    calc_column_density(gridpoint, evalpoint, column_tot, NSPEC-1);

    calc_column_density(gridpoint, evalpoint, column_H, H_nr);
    calc_column_density(gridpoint, evalpoint, column_H2, H2_nr);
    calc_column_density(gridpoint, evalpoint, column_HD, HD_nr);
    calc_column_density(gridpoint, evalpoint, column_C, C_nr);
    calc_column_density(gridpoint, evalpoint, column_CO, CO_nr);


    /* Calculate the visual extinction */

    calc_AV(column_tot, AV);


    /* Calculcate the UV field */

    calc_UV_field(antipod, AV, rad_surface, UV_field);


    /* Calculate the dust temperature */

    calc_temperature_dust(UV_field, rad_surface, temperature_dust);


    /* Calculate the chemical abundances given the current temperatures and radiation field */

    chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
                column_H2, column_HD, column_C, column_CO, v_turb );


  } /* End of chemistry iteration */


  write_abundances("8");


  /* Initialize the level populations with their LTE values */

  double pop[NGRID*TOT_NLEV];                                            /* level population n_i */

  calc_LTE_populations(gridpoint, energy, weight, temperature_gas, pop);

  write_level_populations("0", pop);

  bool no_thermal_balance = true;

  int niterations = 0;





  // for (int chem_iteration=0; chem_iteration<3; chem_iteration++){
  //
  //   calc_column_density(gridpoint, evalpoint, column_H, H_nr);
  //   calc_column_density(gridpoint, evalpoint, column_H2, H2_nr);
  //   calc_column_density(gridpoint, evalpoint, column_HD, HD_nr);
  //   calc_column_density(gridpoint, evalpoint, column_C, C_nr);
  //   calc_column_density(gridpoint, evalpoint, column_CO, CO_nr);
  //
  //   chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
  //              column_H2, column_HD, column_C, column_CO, v_turb );
  //
  // }
  //
  // write_abundances("level1");



  /* Thermal balance iterations */

  while (no_thermal_balance){

    no_thermal_balance = false;

    niterations++;


    /* Calculate column densities */

    calc_column_density(gridpoint, evalpoint, column_H, H_nr);
    calc_column_density(gridpoint, evalpoint, column_H2, H2_nr);
    calc_column_density(gridpoint, evalpoint, column_HD, HD_nr);
    calc_column_density(gridpoint, evalpoint, column_C, C_nr);
    calc_column_density(gridpoint, evalpoint, column_CO, CO_nr);




    /*   CALCULATE CHEMICAL ABUNDANCES                                                           */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Calculate the chemical abundances by solving the rate equations */

    // chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
    //             column_H2, column_HD, column_C, column_CO, v_turb );


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   CALCULATE LEVEL POPULATIONS (ITERATIVELY)                                               */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Calculate level populations for each line producing species */

    level_populations( gridpoint, evalpoint, antipod, irad, jrad, frequency, v_turb,
                       A_coeff, B_coeff, C_coeff, R, pop, dpop, C_data,
                       coltemp, icol, jcol, temperature_gas, temperature_dust,
                       weight, energy, mean_intensity );

    write_level_populations("level1c", pop);

    write_line_intensities("level1c", mean_intensity);

    // return;



    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   CALCULATE HEATING AND COOLING                                                           */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


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


      /* Check for thermal balance (convergence) */

      if (thermal_ratio > THERMAL_PREC){

        no_thermal_balance = true;

        update_temperature_gas(thermal_flux, gridp, temperature_gas, previous_temperature_gas );

      }

    } /* end of gridp loop over grid points */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Limit the number of iterations */

    if (niterations >= MAX_NITERATIONS){

      no_thermal_balance = false;
    }


  } /* end of thermal balance iterations */








  /* Write output */

  string tag = "";

  //
  //
  // write_line_intensities(tag, test_line_datafile, mean_intensity);



  SECTION("Check reading the files"){
    CHECK( 1==1 );
  }


}

/*-----------------------------------------------------------------------------------------------*/
