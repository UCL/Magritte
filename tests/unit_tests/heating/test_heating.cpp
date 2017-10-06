/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_heating:                                                                                 */
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
#include "../../../src/read_input.hpp"
#include "../../../src/species_tools.hpp"
#include "../../../src/read_chemdata.hpp"
#include "../../../src/chemistry.hpp"
#include "../../../src/heating.hpp"




TEST_CASE("Test reaction_rates"){


  /* Since the executables are now in the directory /tests, we have to change the paths */

  string test_grid_inputfile = "../../../" + grid_inputfile;

  string test_spec_datafile  = "../../../" + spec_datafile;

  string test_reac_datafile  = "../../../" + reac_datafile;


  /* Define grid (using types defined in definitions.h)*/

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */


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


  double temperature_gas[NGRID];                          /* temperature of the gas atgrid point */

  initialize_temperature_gas(temperature_gas);

  double temperature_dust[NGRID];                  /* temperature of the dust at each grid point */

  initialize_double_array(temperature_dust, NGRID);

  double rad_surface[NGRID];

  rad_surface[0] = 0.0;

  double AV[NGRID];

  AV[0] = 0.0;

  double UV_field[NGRID];

  UV_field[0] = 0.0;

  double column_H2[NGRID];

  column_H2[0] = 0.0;

  double column_HD[NGRID];

  column_HD[0] = 0.0;

  double column_C[NGRID];

  column_C[0] = 0.0;

  double column_CO[NGRID];

  column_CO[0] = 0.0;


  metallicity = 1.0;

  gas_to_dust = 100.0;

  double v_turb = 0.0;


  long gridp = 0;


  chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
              column_H2, column_HD, column_C, column_CO, v_turb );


  double heating_total = heating( gridpoint, gridp, temperature_gas, temperature_dust,
                                  UV_field, v_turb );

  cout << "Heating total " << heating_total << "\n";


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
