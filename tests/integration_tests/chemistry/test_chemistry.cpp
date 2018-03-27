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
#include <math.h>
#include <omp.h>

#include <iostream>
#include <string>
#include <sstream>

#include "catch.hpp"

#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"

#include "../../../src/initializers.hpp"
#include "../../../src/species_tools.hpp"

#include "../../../src/read_input.hpp"
#include "../../../src/read_chemdata.hpp"

#include "../../../src/create_rays.hpp"
#include "../../../src/ray_tracing.hpp"

#include "../../../src/calc_rad_surface.hpp"
#include "../../../src/calc_column_density.hpp"
#include "../../../src/calc_UV_field.hpp"
#include "../../../src/calc_AV.hpp"
#include "../../../src/calc_temperature_dust.hpp"
#include "../../../src/chemistry.hpp"
#include "../../../src/calc_LTE_populations.hpp"

#include "../../../src/write_output.hpp"



TEST_CASE("Test chemistry"){


  double time_chem = 0.0;


  /* Since the executables are now in the directory /tests, we have to change the paths */

  std::string test_inputfile = "../../../" + inputfile;

  std::string test_spec_datafile  = "../../../" + spec_datafile;

  std::string test_reac_datafile  = "../../../" + reac_datafile;


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


  /* Create the HEALPix vectors */

  double healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  create_rays(healpixvector, antipod);


  /* Ray tracing */

  ray_tracing(healpixvector, cell, evalpoint);



  double AV[NCELLS*NRAYS];

  initialize_double_array(AV, NCELLS*NRAYS);

  double UV_field[NCELLS];

  initialize_double_array(UV_field, NCELLS);

  double column_tot[NCELLS*NRAYS];

  initialize_double_array(column_tot, NCELLS*NRAYS);

  double column_H[NCELLS*NRAYS];

  initialize_double_array(column_H, NCELLS*NRAYS);

  double column_H2[NCELLS*NRAYS];

  initialize_double_array(column_H2, NCELLS*NRAYS);

  double column_HD[NCELLS*NRAYS];

  initialize_double_array(column_HD, NCELLS*NRAYS);

  double column_C[NCELLS*NRAYS];

  initialize_double_array(column_C, NCELLS*NRAYS);

  double column_CO[NCELLS*NRAYS];

  initialize_double_array(column_CO, NCELLS*NRAYS);


  double rad_surface[NCELLS*NRAYS];

  initialize_double_array(rad_surface, NCELLS*NRAYS);


  double G_external[3];                                       /* external radiation field vector */

  G_external[0] = G_EXTERNAL_X;
  G_external[1] = G_EXTERNAL_Y;
  G_external[2] = G_EXTERNAL_Z;

  /* Calculate the radiation surface */

  calc_rad_surface(G_external, healpixvector, rad_surface);

  write_abundances("0");








  /* Calculate column densities */

  calc_column_density(cell, evalpoint, column_tot, NSPEC-1);
  calc_column_density(cell, evalpoint, column_H, nr_H);
  calc_column_density(cell, evalpoint, column_H2, nr_H2);
  calc_column_density(cell, evalpoint, column_HD, nr_HD);
  calc_column_density(cell, evalpoint, column_C, nr_C);
  calc_column_density(cell, evalpoint, column_CO, nr_CO);


  write_double_2("column_tot", "0", NCELLS, NRAYS, column_tot);
  write_double_2("column_density_C","0", NCELLS, NRAYS, column_C);
  write_double_2("column_density_H2", "0", NCELLS, NRAYS, column_H2);
  write_double_2("column_density_CO", "0", NCELLS, NRAYS, column_CO);

  write_eval("0", evalpoint);

  write_rays("", healpixvector);

  /* Calculate the visual extinction */

  calc_AV(column_tot, AV);


  /* Calculcate the UV field */

  calc_UV_field(antipod, AV, rad_surface, UV_field);


  double temperature_gas[NCELLS];

  initialize_temperature_gas(temperature_gas);

  double temperature_dust[NCELLS];

  initialize_double_array(temperature_dust, NCELLS);


  write_UV_field("0", UV_field);

  write_AV("0", AV);

  write_radfield_tools( "0", AV, 1000.0, column_H2, column_CO );

  write_rad_surface("0", rad_surface);

  write_eval("0", evalpoint);


  /* Make aguess for the gas temperature */

  guess_temperature_gas(UV_field, temperature_gas);


  /* Calculate the dust temperature */

  calc_temperature_dust(UV_field, rad_surface, temperature_dust);


  write_temperature_gas("0", temperature_gas);

  write_temperature_dust("0", temperature_dust);




  /* Iterate over the chemistry alone */

  for (int iteration=0; iteration<8; iteration++){


    /* Construct the tags */

    std::stringstream ss;
    ss << iteration + 1;
    std::string tag = ss.str();


    /* Temporary storage for the species */

    SPECIES old_species[NSPEC];

    for (int spec; spec<NSPEC; spec++){

      old_species[spec] = species[spec];
    }


    // species[nr_H2].abn[0] = 0.0;

    /* Calculate column densities */

    calc_column_density(cell, evalpoint, column_tot, NSPEC-1);
    calc_column_density(cell, evalpoint, column_H, nr_H);
    calc_column_density(cell, evalpoint, column_H2, nr_H2);
    calc_column_density(cell, evalpoint, column_HD, nr_HD);
    calc_column_density(cell, evalpoint, column_C, nr_C);
    calc_column_density(cell, evalpoint, column_CO, nr_CO);


    write_double_2("column_tot", tag, NCELLS, NRAYS, column_tot);
    write_double_2("column_density_C", tag, NCELLS, NRAYS, column_C);
    write_double_2("column_density_H2", tag, NCELLS, NRAYS, column_H2);
    write_double_2("column_density_CO", tag, NCELLS, NRAYS, column_CO);



    /* Calculate the visual extinction */

    calc_AV(column_tot, AV);


    /* Calculcate the UV field */

    calc_UV_field(antipod, AV, rad_surface, UV_field);



    write_UV_field(tag, UV_field);

    write_AV(tag, AV);

    write_radfield_tools( tag, AV, 1000.0, column_H2, column_CO );

    write_rad_surface(tag, rad_surface);


    /* Calculate the dust temperature */

    calc_temperature_dust(UV_field, rad_surface, temperature_dust);


    write_temperature_dust(tag, temperature_dust);


    /* Calculate the chemical abundances given the current temperatures and radiation field */

    time_chem -= omp_get_wtime();

    chemistry( cell, temperature_gas, temperature_dust, rad_surface, AV,
               column_H2, column_HD, column_C, column_CO );

    time_chem += omp_get_wtime();

    /* Check for convergence */

    for (int spec=0; spec<NSPEC; spec++){

      double max_difference = 0.0;

      for (long n=0; n<NCELLS; n++){

        double difference = fabs(old_cell[n].abundance[spec] - cell[n].abundance[spec]);

        if (difference > max_difference){

          max_difference = difference;
        }
      }


      // std::cout << "max difference for " << species[spec].sym << " is " << max_difference << "\n";

    }


    write_abundances(tag);

    write_reaction_rates(tag, reaction);


  } /* End of chemistry iteration */

  printf("Time in chemistry is %lE\n", time_chem);

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
