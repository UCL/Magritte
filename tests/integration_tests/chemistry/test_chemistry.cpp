/* Frederik De Ceuster - University College London                                               */
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
#include "../../../src/data_tools.hpp"
#include "../../../src/species_tools.hpp"

#include "../../../src/read_input.hpp"
#include "../../../src/read_chemdata.hpp"

#include "../../../src/create_healpixvectors.hpp"
#include "../../../src/ray_tracing.hpp"

#include "../../../src/rad_surface_calculator.hpp"
#include "../../../src/column_density_calculator.hpp"
#include "../../../src/UV_field_calculator.hpp"
#include "../../../src/AV_calculator.hpp"
#include "../../../src/dust_temperature_calculation.hpp"
#include "../../../src/abundances.hpp"

#include "../../../src/write_output.hpp"



TEST_CASE("Test chemistry"){


  metallicity = 1.0;

  gas_to_dust = 100.0;

  double v_turb = 1.0;


  /* Since the executables are now in the directory /tests, we have to change the paths */

  string test_grid_inputfile = "../../../" + grid_inputfile;

  string test_spec_datafile  = "../../../" + spec_datafile;

  string test_reac_datafile  = "../../../" + reac_datafile;


  /* Define grid (using types defined in definitions.h)*/

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  /* NOTE: gridpoint does not have to be initialized as long as read_input works */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */

  initialize_evalpoint(evalpoint);


  /* Read input file */

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


  /* Create the HEALPix vectors */

  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  create_healpixvectors(unit_healpixvector, antipod);


  /* Ray tracing */

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);


  double temperature_gas[NGRID];

  initialize_temperature_gas(temperature_gas);

  double temperature_dust[NGRID];

  initialize_double_array(temperature_dust, NGRID);


  double AV[NGRID*NRAYS];

  initialize_double_array(AV, NGRID*NRAYS);

  double UV_field[NGRID];

  initialize_double_array(UV_field, NGRID);

  double column_H2[NGRID*NRAYS];

  initialize_double_array(column_H2, NGRID*NRAYS);

  double column_HD[NGRID*NRAYS];

  initialize_double_array(column_HD, NGRID*NRAYS);

  double column_C[NGRID*NRAYS];

  initialize_double_array(column_C, NGRID*NRAYS);

  double column_CO[NGRID*NRAYS];

  initialize_double_array(column_CO, NGRID*NRAYS);


  double rad_surface[NGRID*NRAYS];

  initialize_double_array(rad_surface, NGRID*NRAYS);


  double G_external[3];                                       /* external radiation field vector */

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;

  /* Calculate the radiation surface */

  rad_surface_calculator(G_external, unit_healpixvector, rad_surface);


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


  abundances( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
              column_H2, column_HD, column_C, column_CO, v_turb );





  /* Write the results of the integration */

  FILE *abn_file = fopen("output/abundances_result.txt", "w");

  if (abn_file == NULL){

    printf("Error opening file!\n");
    exit(1);
  }


  for (int spec=0; spec<NSPEC; spec++){

    fprintf( abn_file, "%lE\t", species[spec].abn[0] );
  }


  fclose(abn_file);



  SECTION("Check reading the files"){


    // for(int spec=0; spec<NSPEC; spec++){
    //
    //   printf( "%s\t%.2lE\t%.1lf\n",
    //           species[spec].sym.c_str(), species[spec].abn[0], species[spec].mass );
    // }
    //
    // for(int reac=0; reac<NREAC; reac++){
    //
    //   printf( "%-3s + %-3s + %-3s  ->  %-3s + %-3s + %-3s + %-3s \n"
    //           "with alpha = %-10.2lE, beta = %-10.2lE, gamma = %-10.2lE \t"
    //           "RT_min = %-10.2lE, RT_max = %-10.2lE, duplicates = %d \n",
    //           reaction[reac].R1.c_str(), reaction[reac].R2.c_str(), reaction[reac].R3.c_str(),
    //           reaction[reac].P1.c_str(), reaction[reac].P2.c_str(), reaction[reac].P3.c_str(), reaction[reac].P4.c_str(),
    //           reaction[reac].alpha, reaction[reac].beta, reaction[reac].gamma,
    //           reaction[reac].RT_min, reaction[reac].RT_max,
    //           reaction[reac].dup );
    // }



    CHECK( 1==1 );
  }


}

/*-----------------------------------------------------------------------------------------------*/
