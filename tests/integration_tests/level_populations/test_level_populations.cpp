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

  string test_inputfile = "../../../" + inputfile;

  string test_spec_datafile  = "../../../" + spec_datafile;

  string test_reac_datafile  = "../../../" + reac_datafile;

  string test_line_datafile[NLSPEC];


  for(int l=0; l<NLSPEC; l++){

    test_line_datafile[l] = "../../../" + line_datafile[l];
  }


  /* Define grid (using types defined in definitions.h)*/

  CELL cell[NCELLS];                                                     /* grid points */

  /* NOTE: cell does not have to be initialized as long as read_input works */

  EVALPOINT evalpoint[NCELLS*NCELLS];                     /* evaluation points for each grid point */

  initialize_evalpoint(evalpoint);


  /* Read input file */

  cout << "   ! file :" << inputfile << "\n";

  read_input(test_inputfile, cell);


  /* Read the species (and their initial abundances) */

  read_species(test_spec_datafile);


  /* Get and store the species numbers of some inportant species */

  nr_e    = get_species_nr("e-");                       /* species nr corresponding to electrons */

  nr_H2   = get_species_nr("H2");                              /* species nr corresponding to H2 */

  nr_HD   = get_species_nr("HD");                              /* species nr corresponding to HD */

  nr_C    = get_species_nr("C");                                /* species nr corresponding to C */

  nr_H    = get_species_nr("H");                                /* species nr corresponding to H */

  nr_H2x  = get_species_nr("H2+");                            /* species nr corresponding to H2+ */

  nr_HCOx = get_species_nr("HCO+");                          /* species nr corresponding to HCO+ */

  nr_H3x  = get_species_nr("H3+");                            /* species nr corresponding to H3+ */

  nr_H3Ox = get_species_nr("H3O+");                          /* species nr corresponding to H3O+ */

  nr_Hex  = get_species_nr("He+");                            /* species nr corresponding to He+ */

  nr_CO   = get_species_nr("CO");                              /* species nr corresponding to CO */


  /* Read the reactions */

  read_reactions(test_reac_datafile);


  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NCELLS*NCELLS);

  initialize_long_array(raytot, NCELLS*NRAYS);

  initialize_long_array(cum_raytot, NCELLS*NRAYS);


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

  double R[NCELLS*TOT_NLEV2];                                           /* transition matrix R_ij */

  initialize_double_array(R, NCELLS*TOT_NLEV2);


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

  ray_tracing(healpixvector, cell, evalpoint);


  double G_external[3];                                       /* external radiation field vector */

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;


  double rad_surface[NCELLS*NRAYS];

  initialize_double_array(rad_surface, NCELLS*NRAYS);


  /* Calculate the radiation surface */

  calc_rad_surface(G_external, healpixvector, rad_surface);



  double temperature_dust[NCELLS];                  /* temperature of the dust at each grid point */

  initialize_double_array(temperature_dust, NCELLS);

  double column_tot[NCELLS*NRAYS];

  initialize_double_array(column_tot, NCELLS*NRAYS);

  double column_H[NCELLS*NRAYS];                  /* H column density for each ray and grid point */

  initialize_double_array(column_H, NCELLS*NRAYS);

  double column_H2[NCELLS*NRAYS];                /* H2 column density for each ray and grid point */

  initialize_double_array(column_H2, NCELLS*NRAYS);

  double column_HD[NCELLS*NRAYS];                /* HD column density for each ray and grid point */

  initialize_double_array(column_HD, NCELLS*NRAYS);

  double column_C[NCELLS*NRAYS];                  /* C column density for each ray and grid point */

  initialize_double_array(column_C, NCELLS*NRAYS);

  double column_CO[NCELLS*NRAYS];                /* CO column density for each ray and grid point */

  initialize_double_array(column_CO, NCELLS*NRAYS);

  double AV[NCELLS*NRAYS];                       /* Visual extinction (only takes into account H) */

  initialize_double_array(AV, NCELLS*NRAYS);

  double UV_field[NCELLS];

  initialize_double_array(UV_field, NCELLS);

  double dpop[NCELLS*TOT_NLEV];        /* change in level population n_i w.r.t previous iteration */

  initialize_double_array(dpop, NCELLS*TOT_NLEV);

  double mean_intensity[NCELLS*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NCELLS*TOT_NRAD);


  /* Make a guess for the gas temperature, based on he UV field */

  calc_column_density(cell, evalpoint, column_tot, NSPEC-1);

  calc_AV(column_tot, AV);

  calc_UV_field(antipod, AV, rad_surface, UV_field);

  double temperature_gas[NCELLS];                    /* temperature of the gas at each grid point */

  guess_temperature_gas(UV_field, temperature_gas);

  double previous_temperature_gas[NCELLS];    /* temp. of gas at each grid point, prev. iteration */

  initialize_previous_temperature_gas(previous_temperature_gas, temperature_gas);


  /* Preliminary chemistry iterations */

  for (int chem_iteration=0; chem_iteration<5; chem_iteration++){


    /* Calculate column densities */

    calc_column_density(cell, evalpoint, column_tot, NSPEC-1);

    calc_column_density(cell, evalpoint, column_H, nr_H);
    calc_column_density(cell, evalpoint, column_H2, nr_H2);
    calc_column_density(cell, evalpoint, column_HD, nr_HD);
    calc_column_density(cell, evalpoint, column_C, nr_C);
    calc_column_density(cell, evalpoint, column_CO, nr_CO);


    /* Calculate the visual extinction */

    calc_AV(column_tot, AV);


    /* Calculcate the UV field */

    calc_UV_field(antipod, AV, rad_surface, UV_field);


    /* Calculate the dust temperature */

    calc_temperature_dust(UV_field, rad_surface, temperature_dust);


    /* Calculate the chemical abundances given the current temperatures and radiation field */

    chemistry( cell, temperature_gas, temperature_dust, rad_surface, AV,
               column_H2, column_HD, column_C, column_CO );


  } /* End of chemistry iteration */


  write_abundances("8");


  /* Initialize the level populations with their LTE values */

  double pop[NCELLS*TOT_NLEV];                                            /* level population n_i */

  // calc_LTE_populations(cell, energy, weight, temperature_gas, pop);

  write_level_populations("0", pop);


  bool somewhere_no_thermal_balance_iteration = true;

  bool no_thermal_balance_iteration[NCELLS];

  initialize_bool(true, no_thermal_balance_iteration, NCELLS);

  int niterations = 0;



  /* Thermal balance iterations */

  while (somewhere_no_thermal_balance_iteration){

    somewhere_no_thermal_balance_iteration = false;

    niterations++;



    /*   CALCULATE CHEMICAL ABUNDANCES                                                           */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    for (int chem_iteration=0; chem_iteration<3; chem_iteration++){


      /* Calculate column densities */

      calc_column_density(cell, evalpoint, column_H, nr_H);
      calc_column_density(cell, evalpoint, column_H2, nr_H2);
      calc_column_density(cell, evalpoint, column_HD, nr_HD);
      calc_column_density(cell, evalpoint, column_C, nr_C);
      calc_column_density(cell, evalpoint, column_CO, nr_CO);


      /* Calculate the chemical abundances given the current temperatures and radiation field */

      chemistry( cell, temperature_gas, temperature_dust, rad_surface, AV,
                 column_H2, column_HD, column_C, column_CO );


    } /* End of chemistry iteration */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




    /*   CALCULATE LEVEL POPULATIONS (ITERATIVELY)                                               */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Initialize the level populations to their LTE values */

    calc_LTE_populations(cell, energy, weight, temperature_gas, pop);


    /* Calculate level populations for each line producing species */

    level_populations( cell, evalpoint, antipod, irad, jrad, frequency,
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


    double heating_total[NCELLS];

    double heating_1[NCELLS];
    double heating_2[NCELLS];
    double heating_3[NCELLS];
    double heating_4[NCELLS];
    double heating_5[NCELLS];
    double heating_6[NCELLS];
    double heating_7[NCELLS];
    double heating_8[NCELLS];
    double heating_9[NCELLS];
    double heating_10[NCELLS];
    double heating_11[NCELLS];


    double cooling_total[NCELLS];

    // calc_LTE_populations(cell, energy, weight, temperature_gas, pop);

    int nr_can_reac = 0;
    int nr_can_phot = 0;
    int nr_all = 0;


    int canonical_reactions[NREAC];
    int can_photo_reactions[NREAC];
    int all_reactions[NREAC];

    calc_column_density(cell, evalpoint, column_H, nr_H);
    calc_column_density(cell, evalpoint, column_H2, nr_H2);
    calc_column_density(cell, evalpoint, column_HD, nr_HD);
    calc_column_density(cell, evalpoint, column_C, nr_C);
    calc_column_density(cell, evalpoint, column_CO, nr_CO);


    write_reaction_rates("preheat", reaction);

    write_abundances("preheat");


    /* Calculate the thermal balance for each cell */

    for (long o=0; o<NCELLS; o++){

      if (no_thermal_balance_iteration[o]){

        no_thermal_balance_iteration[o] = false;

        double heating_components[12];

        nr_can_reac = 0;
        nr_can_phot = 0;
        nr_all = 0;

        reaction_rates( temperature_gas, temperature_dust, rad_surface, AV,
                        column_H2, column_HD, column_C, column_CO, o,
                        &nr_can_reac, canonical_reactions,
                        &nr_can_phot, can_photo_reactions,
                        &nr_all, all_reactions );


        heating_total[o] = heating( cell, o, temperature_gas, temperature_dust,
                                        UV_field, heating_components );

        cooling_total[o] = cooling( o, irad, jrad, A_coeff, B_coeff, frequency, weight,
                                        pop, mean_intensity );


        heating_1[o] = heating_components[0];
        heating_2[o] = heating_components[1];
        heating_3[o] = heating_components[2];
        heating_4[o] = heating_components[3];
        heating_5[o] = heating_components[4];
        heating_6[o] = heating_components[5];
        heating_7[o] = heating_components[6];
        heating_8[o] = heating_components[7];
        heating_9[o] = heating_components[8];
        heating_10[o] = heating_components[9];
        heating_11[o] = heating_components[10];
        // heating_total[o] = heating_components[11];


        double thermal_flux = heating_total[o] - cooling_total[o];

        double thermal_ratio = 0.0;

        if( fabs(heating_total[o] + cooling_total[o]) > 0.0 ){

          thermal_ratio = 2.0 * fabs(thermal_flux)
                          / fabs(heating_total[o] + cooling_total[o]);
        }


        /* Check for thermal balance (convergence) */

        if (thermal_ratio > THERMAL_PREC){

          no_thermal_balance_iteration[o]    = true;

          somewhere_no_thermal_balance_iteration = true;

          update_temperature_gas(thermal_flux, o, temperature_gas, previous_temperature_gas );

        }


      } /* end of if no thermal balance */

    } /* end of o loop over grid points */


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/



    string name = species[lspec_nr[0]].sym;

    string file_name ="cool_" + name;

    write_double_1(file_name, "level1c", NCELLS, cooling_total);


    write_double_1("heat_1", "level1c", NCELLS, heating_1);
    write_double_1("heat_2", "level1c", NCELLS, heating_2);
    write_double_1("heat_3", "level1c", NCELLS, heating_3);
    write_double_1("heat_4", "level1c", NCELLS, heating_4);
    write_double_1("heat_5", "level1c", NCELLS, heating_5);
    write_double_1("heat_6", "level1c", NCELLS, heating_6);
    write_double_1("heat_7", "level1c", NCELLS, heating_7);
    write_double_1("heat_8", "level1c", NCELLS, heating_8);
    write_double_1("heat_9", "level1c", NCELLS, heating_9);
    write_double_1("heat_10", "level1c", NCELLS, heating_10);
    write_double_1("heat_11", "level1c", NCELLS, heating_11);

    write_double_1("heat", "level1c", NCELLS, heating_total);



    /* Limit the number of iterations */

    if (niterations >= MAX_NITERATIONS){

      somewhere_no_thermal_balance_iteration = false;
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
