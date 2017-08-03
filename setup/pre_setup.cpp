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

int main(){


  cout << "                    \n";
  cout << "Pre-setup for 3D-RT \n";
  cout << "------------------- \n\n";


  /* Get nlspec from line 32 in parameters.txt */

  int nlspec = get_nr(32);


  /* Write NLSPEC file */

  FILE *nlspec_file = fopen("NLSPEC.hpp", "w");

  fprintf( nlspec_file, "#define NLSPEC %d", nlspec );

  fclose(nlspec_file);

  cout << "(pre-setup): NLSPEC : " << nlspec << "\n\n";


  cout << "(pre-setup): done, setup can now be compiled \n\n";

}

/*-----------------------------------------------------------------------------------------------*/
