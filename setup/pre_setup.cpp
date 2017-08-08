/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* pre_setup: get the number of line species to make the setup file                              */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <iostream>
using namespace std;

#include "setup_tools.cpp"



/* main: Sets up the definitions.hpp file                                                        */
/*-----------------------------------------------------------------------------------------------*/

int main(int argc, char *argv[]){


  cout << "                    \n";
  cout << "Pre-setup for 3D-RT \n";
  cout << "------------------- \n\n";


  string parameters_file = argv[1];


  /* Get nlspec from line 34 in PARAMETERS_FILE */

  int nlspec = get_nr(parameters_file, 34);


  /* Write NLSPEC file */

  FILE *psp_file = fopen("pre_setup_parameters.hpp", "w");

  fprintf( psp_file, "#define PARAMETERS_FILE \"%s\" \n", parameters_file.c_str() );
  fprintf( psp_file, "#define NLSPEC %d", nlspec );

  fclose(psp_file);


  cout << "(pre-setup): PARAMETERS_FILE = " << parameters_file << "\n";

  cout << "(pre-setup): NLSPEC : " << nlspec << "\n\n";


  cout << "(pre-setup): done, setup can now be compiled \n\n";

}

/*-----------------------------------------------------------------------------------------------*/
