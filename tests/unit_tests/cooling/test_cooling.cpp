/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_chemistry:                                                                               */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"
#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/read_linedata.hpp"
#include "../../../src/create_rays.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/data_tools.hpp"
#include "../../../src/setup_data_structures.hpp"
#include "../../../src/species_tools.hpp"
#include "../../../src/read_chemdata.hpp"
#include "../../../src/level_populations.hpp"
#include "../../../src/cooling.hpp"
#include "../../../src/write_output.hpp"





TEST_CASE("Test chemistry"){



  /* Since the executables are now in the directory /tests, we have to change the paths */

  string test_inputfile = "../../../" + inputfile;

  string test_spec_datafile  = "../../../" + spec_datafile;

  string test_reac_datafile  = "../../../" + reac_datafile;

  string test_line_datafile[NLSPEC];

  test_line_datafile[0] = "../../../" + line_datafile[0];


  /* Define grid (using types defined in definitions.h)*/

  CELL cell[NCELLS];                                                     /* grid points */

  EVALPOINT evalpoint[NCELLS*NCELLS];                     /* evaluation points for each grid point */

  initialize_evalpoint(evalpoint);


  /* Read input file */

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

  create_rays(healpixvector, antipod);


  /* Execute ray_tracing */

  ray_tracing(healpixvector, cell, evalpoint);


  double temperature_gas[NCELLS];                    /* temperature of the gas at each grid point */

  initialize_temperature_gas(temperature_gas);

  double pop[NCELLS*TOT_NLEV];                                            /* level population n_i */

  initialize_level_populations(energy, temperature_gas, pop);

  double dpop[NCELLS*TOT_NLEV];        /* change in level population n_i w.r.t previous iteration */

  initialize_double_array(dpop, NCELLS*TOT_NLEV);

  double mean_intensity[NCELLS*TOT_NRAD];                             /* mean intensity for a ray */

  initialize_double_array(mean_intensity, NCELLS*TOT_NRAD);


  for (int lspec=0; lspec<NLSPEC; lspec++){

    for (long n=0; n<NCELLS; n++){

      for (int i=0; i<nlev[lspec]; i++){

        printf("%lE\n", pop[LSPECGRIDLEV(lspec,n, i)]);
      }
    }
  }



  /* Calculate level populations for each line producing species */

  level_populations( antipod, cell, evalpoint, irad, jrad, frequency,
                     A_coeff, B_coeff, C_coeff, R, pop, dpop, C_data,
                     coltemp, icol, jcol, temperature_gas, weight, energy, mean_intensity );


  for (int lspec=0; lspec<NLSPEC; lspec++){

    for (long n=0; n<NCELLS; n++){

      for (int i=0; i<nlev[lspec]; i++){

        printf("%lE\n", pop[LSPECGRIDLEV(lspec,n, i)]);
      }
    }
  }


  for (long o=0; o<NCELLS; o++){

    double cooling_total = cooling( o, irad, jrad, A_coeff, B_coeff, frequency,
                                    pop, mean_intensity );

    cout << "Coolimg " << cooling_total << "\n";


  } /* end of o loop over grid points */



  CHECK( 1==1 );


  /* Write output */

  write_output(healpixvector, antipod, cell, evalpoint, pop, weight, energy);



}

/*-----------------------------------------------------------------------------------------------*/
