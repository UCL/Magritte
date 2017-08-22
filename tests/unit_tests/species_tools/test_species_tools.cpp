/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_species_tools: tests the tools to handle SPECIES objects as defined in species_tools     */
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
#include "../../../src/species_tools.hpp"



TEST_CASE("Test species_tools"){


  species[0].sym = "dummy";
  species[1].sym = "H";
  species[2].sym = "H2";
  species[3].sym = "He";
  species[4].sym = "Li";
  species[5].sym = "e-";



  CHECK( get_canonical_name("e") == "e-" );

  CHECK( get_species_nr("e") == 5 );

  CHECK( get_species_nr("Li") == 4 );

  CHECK( get_species_nr("oH2") == 2 );

  CHECK( get_species_nr("pH2") == 2 );

  CHECK( get_species_nr("o-H2") == 2 );

  CHECK( get_species_nr("p-H2") == 2 );

  char list[5];


  list[0] = check_ortho_para("B");

  // cout << get_species_nr("e") << "\n";

}

/*-----------------------------------------------------------------------------------------------*/
