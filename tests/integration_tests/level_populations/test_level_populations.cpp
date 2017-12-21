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
#include "../../../src/reaction_rates.hpp"
#include "../../../src/chemistry.hpp"
#include "../../../src/calc_LTE_populations.hpp"
#include "../../../src/level_populations.hpp"
#include "../../../src/heating.hpp"
#include "../../../src/cooling.hpp"
#include "../../../src/update_temperature_gas.hpp"

#include "../../../src/write_output.hpp"





TEST_CASE("Test level populations"){


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

  double healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  create_healpixvectors(healpixvector, antipod);


  /* Execute ray_tracing */

  ray_tracing(healpixvector, gridpoint, evalpoint);


  double G_external[3];                                       /* external radiation field vector */

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  double rad_surface[NGRID*NRAYS];

  initialize_double_array(rad_surface, NGRID*NRAYS);


  /* Calculate the radiation surface */

  calc_rad_surface(G_external, healpixvector, rad_surface);



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

  for (int chem_iteration=0; chem_iteration<5; chem_iteration++){


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
               column_H2, column_HD, column_C, column_CO );


  } /* End of chemistry iteration */


  write_abundances("8");


  /* Initialize the level populations with their LTE values */

  double pop[NGRID*TOT_NLEV];                                            /* level population n_i */

  // calc_LTE_populations(gridpoint, energy, weight, temperature_gas, pop);

  write_level_populations("0", pop);


  bool somewhere_no_thermal_balance = true;

  bool no_thermal_balance[NGRID];

  initialize_bool(true, NGRID, no_thermal_balance);

  int niterations = 0;



  /* Thermal balance iterations */

  while (somewhere_no_thermal_balance){

    somewhere_no_thermal_balance = false;

    niterations++;



    /*   CALCULATE CHEMICAL ABUNDANCES                                                           */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    for (int chem_iteration=0; chem_iteration<3; chem_iteration++){


      /* Calculate column densities */

      calc_column_density(gridpoint, evalpoint, column_H, H_nr);
      calc_column_density(gridpoint, evalpoint, column_H2, H2_nr);
      calc_column_density(gridpoint, evalpoint, column_HD, HD_nr);
      calc_column_density(gridpoint, evalpoint, column_C, C_nr);
      calc_column_density(gridpoint, evalpoint, column_CO, CO_nr);


      /* Calculate the chemical abundances given the current temperatures and radiation field */

      chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
                 column_H2, column_HD, column_C, column_CO );


    } /* End of chemistry iteration */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   CALCULATE LEVEL POPULATIONS (ITERATIVELY)                                               */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Initialize the level populations to their LTE values */

    calc_LTE_populations(gridpoint, energy, weight, temperature_gas, pop);


    /* Calculate level populations for each line producing species */

    level_populations( gridpoint, evalpoint, antipod, irad, jrad, frequency,
                       A_coeff, B_coeff, C_coeff, R, pop, dpop, C_data,
                       coltemp, icol, jcol, temperature_gas, temperature_dust,
                       weight, energy, mean_intensity );

    //
    // write_level_populations("level1c", pop);
    //
    // write_line_intensities("level1c", mean_intensity);
    //
    // write_transition_levels("", irad, jrad);


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   CALCULATE HEATING AND COOLING                                                           */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    double heating_total[NGRID];

    double heating_1[NGRID];
    double heating_2[NGRID];
    double heating_3[NGRID];
    double heating_4[NGRID];
    double heating_5[NGRID];
    double heating_6[NGRID];
    double heating_7[NGRID];
    double heating_8[NGRID];
    double heating_9[NGRID];
    double heating_10[NGRID];
    double heating_11[NGRID];


    double cooling_total[NGRID];

    // calc_LTE_populations(gridpoint, energy, weight, temperature_gas, pop);

    int nr_can_reac = 0;
    int nr_can_phot = 0;
    int nr_all = 0;


    int canonical_reactions[NREAC];
    int can_photo_reactions[NREAC];
    int all_reactions[NREAC];

    calc_column_density(gridpoint, evalpoint, column_H, H_nr);
    calc_column_density(gridpoint, evalpoint, column_H2, H2_nr);
    calc_column_density(gridpoint, evalpoint, column_HD, HD_nr);
    calc_column_density(gridpoint, evalpoint, column_C, C_nr);
    calc_column_density(gridpoint, evalpoint, column_CO, CO_nr);


    write_reaction_rates("preheat", reaction);

    write_abundances("preheat");


    /* Calculate the thermal balance for each gridpoint */

    for (long gridp=0; gridp<NGRID; gridp++){

      if (no_thermal_balance[gridp]){

        no_thermal_balance[gridp] = false;

        double heating_components[12];

        nr_can_reac = 0;
        nr_can_phot = 0;
        nr_all = 0;

        reaction_rates( temperature_gas, temperature_dust, rad_surface, AV,
                        column_H2, column_HD, column_C, column_CO, gridp,
                        &nr_can_reac, canonical_reactions,
                        &nr_can_phot, can_photo_reactions,
                        &nr_all, all_reactions );


        heating_total[gridp] = heating( gridpoint, gridp, temperature_gas, temperature_dust,
                                        UV_field, heating_components );

        cooling_total[gridp] = cooling( gridp, irad, jrad, A_coeff, B_coeff, frequency, weight,
                                        pop, mean_intensity );


        heating_1[gridp] = heating_components[0];
        heating_2[gridp] = heating_components[1];
        heating_3[gridp] = heating_components[2];
        heating_4[gridp] = heating_components[3];
        heating_5[gridp] = heating_components[4];
        heating_6[gridp] = heating_components[5];
        heating_7[gridp] = heating_components[6];
        heating_8[gridp] = heating_components[7];
        heating_9[gridp] = heating_components[8];
        heating_10[gridp] = heating_components[9];
        heating_11[gridp] = heating_components[10];
        // heating_total[gridp] = heating_components[11];


        double thermal_flux = heating_total[gridp] - cooling_total[gridp];

        double thermal_ratio = 0.0;

        if( fabs(heating_total[gridp] + cooling_total[gridp]) > 0.0 ){

          thermal_ratio = 2.0 * fabs(thermal_flux)
                          / fabs(heating_total[gridp] + cooling_total[gridp]);
        }


        /* Check for thermal balance (convergence) */

        if (thermal_ratio > THERMAL_PREC){

          no_thermal_balance[gridp]    = true;

          somewhere_no_thermal_balance = true;

          update_temperature_gas(thermal_flux, gridp, temperature_gas, previous_temperature_gas );

        }


      } /* end of if no thermal balance */

    } /* end of gridp loop over grid points */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/



    string name = species[lspec_nr[0]].sym;

    string file_name ="cool_" + name;

    write_double_1(file_name, "level1c", NGRID, cooling_total);


    write_double_1("heat_1", "level1c", NGRID, heating_1);
    write_double_1("heat_2", "level1c", NGRID, heating_2);
    write_double_1("heat_3", "level1c", NGRID, heating_3);
    write_double_1("heat_4", "level1c", NGRID, heating_4);
    write_double_1("heat_5", "level1c", NGRID, heating_5);
    write_double_1("heat_6", "level1c", NGRID, heating_6);
    write_double_1("heat_7", "level1c", NGRID, heating_7);
    write_double_1("heat_8", "level1c", NGRID, heating_8);
    write_double_1("heat_9", "level1c", NGRID, heating_9);
    write_double_1("heat_10", "level1c", NGRID, heating_10);
    write_double_1("heat_11", "level1c", NGRID, heating_11);

    write_double_1("heat", "level1c", NGRID, heating_total);



    /* Limit the number of iterations */

    if (niterations >= MAX_NITERATIONS){

      somewhere_no_thermal_balance = false;
    }


  } /* end of thermal balance iterations */




  write_temperature_gas("final", temperature_gas);

  write_temperature_dust("final", temperature_dust);



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
