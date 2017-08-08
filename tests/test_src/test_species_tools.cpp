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

#include "../../src/declarations.hpp"
#include "../../src/definitions.hpp"
#include "../../src/species_tools.hpp"



TEST_CASE("Test species_tools"){

  species[0].sym = "H";
  species[1].sym = "H2";
  species[2].sym = "He";
  species[3].sym = "Li";
  species[4].sym = "e-";



  CHECK( get_canonical_name("e") == "e-" );

  CHECK( get_species_nr("e") == 4 );

  CHECK( get_species_nr("Li") == 3 );

  CHECK( get_species_nr("oH2") == 1 );

  CHECK( get_species_nr("pH2") == 1 );

  CHECK( get_species_nr("o-H2") == 1 );

  CHECK( get_species_nr("p-H2") == 1 );

  char *list;
  list = (char*) malloc( 5*sizeof(char) );

  list[0] = check_ortho_para("B");

  cout << list[0] << "\n";

}

/*-----------------------------------------------------------------------------------------------*/
