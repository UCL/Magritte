/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Setup: Read the sizes of the datafile and use these in definitions.hpp                        */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <iostream>
using namespace std;

#include "pre_setup_declarations.hpp"
#include "pre_setup_definitions.hpp"
#include "setup_tools.hpp"
#include "../src/setup_data_structures.cpp"
#include "../src/data_tools.cpp"



/* main: Sets up the definitions.hpp file                                                        */
/*-----------------------------------------------------------------------------------------------*/

int main(){


  cout << "                \n";
  cout << "Setup for 3D-RT \n";
  cout << "--------------- \n\n";





  /*   READ PARAMETERS                                                                           */
  /*_____________________________________________________________________________________________*/


  cout << "(setup): reading the parameters.txt file \n";


  /* Get nrays from line 11 in parameters.txt */

  long nrays = get_nr(11);

  long nsides = (long) sqrt(nrays/12.0);



  /* Get theta_crit from line 13 in parameters.txt */

  double theta_crit = get_nr(13);



  /* Get ray_separation2 from line 15 in parameters.txt */

  double ray_separation2 = get_nr(15);



  /* Get sobolev from line 17 in parameters.txt */

  string sobolev = get_string(17);


  /* Get field form from line 19 in parameters.txt */

  string field_form = get_string(19);


  /* Get the grid input file from line 27 in parameters.txt */

  string grid_inputfile = get_file(27);



  /* Get the number of grid points from the input file */

  long ngrid = get_NGRID(grid_inputfile);



  /* Get the species data file from line 29 in parameters.txt */

  string spec_datafile = get_file(29);



  /* Get the number of species from the species data file */

  int nspec = get_NSPEC(spec_datafile);



  /* Get the reaction data file from line 31 in parameters.txt */

  string reac_datafile = get_file(31);



  /* Get the line data files starting from line 37 in parameters.txt */

  for (int l=0; l<NLSPEC; l++){

    line_datafile[l] = get_file(37+2*l);
  }



  /* Get the number of reactions from the reaction data file */

  int nreac = get_NREAC(reac_datafile);


  cout << "(setup): parameters.txt file read \n\n";


  cout << "(setup): PARAMETERS: \n";
  cout << "(setup): nrays           : " << nrays << "\n";
  cout << "(setup): theta_crit      : " << theta_crit << "\n";
  cout << "(setup): ray_separation2 : " << ray_separation2 << "\n";
  cout << "(setup): sobolev         : " << sobolev << "\n";
  cout << "(setup): field_form      : " << field_form << "\n";

  cout << "(setup): grid file       : " << grid_inputfile << "\n";
  cout << "(setup): species file    : " << spec_datafile << "\n";
  cout << "(setup): reactions file  : " << reac_datafile << "\n";

  cout << "(setup): NLSPEC          : " << NLSPEC << "\n";

  for (int l=0; l<NLSPEC; l++){

    cout << "(setup): line file " << l << "     : " << line_datafile[l] << "\n";
  }

  cout << "(setup): ngrid           : " << ngrid << "\n";
  cout << "(setup): nsides          : " << nsides << " ( = sqrt(nrays/12) ) \n";
  cout << "(setup): nspec           : " << nspec << "\n\n";


  /*_____________________________________________________________________________________________*/





  /*   EXTRACT PARAMETERS FROM THE LINE DATA                                                     */
  /*_____________________________________________________________________________________________*/


  cout << "(setup): extracting parameters from line data \n";



  /* Setup data structures */


  nlev = (int*) malloc( NLSPEC*sizeof(double) );
  nrad = (int*) malloc( NLSPEC*sizeof(double) );

  cum_nlev = (int*) malloc( NLSPEC*sizeof(double) );
  cum_nlev2 = (int*) malloc( NLSPEC*sizeof(double) );
  cum_nrad = (int*) malloc( NLSPEC*sizeof(double) );

  ncolpar = (int*) malloc( NLSPEC*sizeof(double) );
  cum_ncolpar = (int*) malloc( NLSPEC*sizeof(double) );



  setup_data_structures1();


  int tot_nlev = cum_nlev[NLSPEC-1] + nlev[NLSPEC-1];                      /* tot. nr. of levels */

  int tot_nrad = cum_nrad[NLSPEC-1] + nrad[NLSPEC-1];                 /* tot. nr. of transitions */

  int tot_nlev2 = cum_nlev2[NLSPEC-1] + nlev[NLSPEC-1]*nlev[NLSPEC-1];
                                                               /* tot of squares of nr of levels */

  int tot_ncolpar = cum_ncolpar[NLSPEC-1] + ncolpar[NLSPEC-1];



  ncoltemp = (int*) malloc( tot_ncolpar*sizeof(double) );
  ncoltran = (int*) malloc( tot_ncolpar*sizeof(double) );
  cum_ncoltemp = (int*) malloc( tot_ncolpar*sizeof(double) );
  cum_ncoltran = (int*) malloc( tot_ncolpar*sizeof(double) );
  tot_ncoltemp = (int*) malloc( NLSPEC*sizeof(double) );
  tot_ncoltran = (int*) malloc( NLSPEC*sizeof(double) );
  cum_tot_ncoltemp = (int*) malloc( NLSPEC*sizeof(double) );
  cum_tot_ncoltran = (int*) malloc( NLSPEC*sizeof(double) );
  cum_ncoltrantemp = (int*) malloc( tot_ncolpar*sizeof(double) );
  tot_ncoltrantemp = (int*) malloc( NLSPEC*sizeof(double) );
  cum_tot_ncoltrantemp = (int*) malloc( NLSPEC*sizeof(double) );



  setup_data_structures2();


  int tot_cum_tot_ncoltran = cum_tot_ncoltran[NLSPEC-1] + tot_ncoltran[NLSPEC-1];
                                                         /* total over the line prodcing species */
  int tot_cum_tot_ncoltemp = cum_tot_ncoltemp[NLSPEC-1] + tot_ncoltemp[NLSPEC-1];
                                                         /* total over the line prodcing species */
  int tot_cum_tot_ncoltrantemp = cum_tot_ncoltrantemp[NLSPEC-1]
                                   + tot_ncoltrantemp[NLSPEC-1];
                                                         /* total over the line prodcing species */


  cout << "(setup): parameters from line data extracted \n\n";


  /*_____________________________________________________________________________________________*/





  /*   WRITE DECLARATIONS                                                                        */
  /*_____________________________________________________________________________________________*/


  cout << "(setup): setting up declarations.hpp \n";


  char buffer1[BUFFER_SIZE];
  char buffer2[BUFFER_SIZE];



  FILE *dec_new = fopen("../src/declarations.hpp", "w");



  /* Write the header */

  FILE *dec_head = fopen("declarations_hd.txt", "r");


  while ( !feof(dec_head) ){

    fgets(buffer1, BUFFER_SIZE, dec_head);

    fprintf(dec_new, "%s", buffer1);
  }

  fclose(dec_head);



  /* Subtract the "../" part of the filenames */

  grid_inputfile = grid_inputfile.erase(0,3);
  spec_datafile  = spec_datafile.erase(0,3);
  reac_datafile  = reac_datafile.erase(0,3);

  for (int l=0; l<NLSPEC; l++){

    line_datafile[l]  = line_datafile[l].erase(0,3);
  }



  /* write the new declarations.hpp part */

  fprintf( dec_new, "#define GRID_INPUTFILE \"%s\" \n\n", grid_inputfile.c_str() );

  fprintf( dec_new, "#define SPEC_DATAFILE  \"%s\" \n\n", spec_datafile.c_str() );

  fprintf( dec_new, "#define REAC_DATAFILE  \"%s\" \n\n", reac_datafile.c_str() );

  for (int l=0; l<NLSPEC; l++){

    fprintf( dec_new, "#define LINE_DATAFILE%d  \"%s\" \n\n", l, line_datafile[l].c_str() );
  }

  fprintf( dec_new, "#define NGRID %ld \n\n", ngrid );

  fprintf( dec_new, "#define NSIDES %ld \n\n", nsides );

  fprintf( dec_new, "#define THETA_CRIT %lf \n\n", theta_crit );

  fprintf( dec_new, "#define RAY_SEPARATION2 %lf \n\n", ray_separation2 );

  fprintf( dec_new, "#define SOBOLEV %s \n\n", sobolev.c_str() );

  fprintf( dec_new, "#define FIELD_FORM \"%s\" \n\n", field_form.c_str() );

  fprintf( dec_new, "#define NSPEC %d \n\n", nspec );

  fprintf( dec_new, "#define NREAC %d \n\n", nreac );

  fprintf( dec_new, "#define NLSPEC %d \n\n", NLSPEC );

  fprintf( dec_new, "#define TOT_NLEV %d \n\n", tot_nlev );

  fprintf( dec_new, "#define TOT_NRAD %d \n\n", tot_nrad );

  fprintf( dec_new, "#define TOT_NLEV2 %d \n\n", tot_nlev2 );

  fprintf( dec_new, "#define TOT_NCOLPAR %d \n\n", tot_ncolpar );

  fprintf( dec_new, "#define TOT_CUM_TOT_NCOLTRAN %d \n\n", tot_cum_tot_ncoltran );

  fprintf( dec_new, "#define TOT_CUM_TOT_NCOLTEMP %d \n\n", tot_cum_tot_ncoltemp );

  fprintf( dec_new, "#define TOT_CUM_TOT_NCOLTRANTEMP %d \n\n", tot_cum_tot_ncoltrantemp );




  /* Write the standard part of definitions */

  FILE *dec_std = fopen("declarations_std.txt", "r");


  while ( !feof(dec_std) ){

    fgets(buffer1, BUFFER_SIZE, dec_std);

    fprintf(dec_new, "%s", buffer1);
  }

  fclose(dec_std);


  fclose(dec_new);


  cout << "(setup): declarations.hpp is set up \n\n";


  /*_____________________________________________________________________________________________*/





  /*   WRITE DEFINITIONS                                                                         */
  /*_____________________________________________________________________________________________*/


  cout << "(setup): setting up definitions.hpp \n";


  FILE *def_new = fopen("../src/definitions.hpp", "w");



  /* Write the header */

  FILE *def_head = fopen("definitions_hd.txt", "r");


  while ( !feof(def_head) ){

    fgets(buffer1, BUFFER_SIZE, def_head);

    fprintf(def_new, "%s", buffer1);
  }

  fclose(def_head);


  /* write the new definitions.hpp part */

  if (NLSPEC == 1){

    fprintf( def_new, "string line_datafile[NLSPEC] = { LINE_DATAFILE0 }; \n \n\n" );
  }

  else{
    fprintf( def_new, "string line_datafile[NLSPEC] = { LINE_DATAFILE0, \\\n" );

    for (int l=1; l<NLSPEC-1; l++){

      line_datafile[l]  = line_datafile[l].erase(0,3);   /* Subtract "../" part of the filenames */

      fprintf( def_new, "                                 LINE_DATAFILE%d, \\\n", l );
    }

    fprintf( def_new, "                                 LINE_DATAFILE%d  }; \n \n\n ", NLSPEC-1 );

  }


  /* Write the standard part of definitions */

  FILE *def_std = fopen("definitions_std.txt", "r");


  while ( !feof(def_std) ){

    fgets(buffer1, BUFFER_SIZE, def_std);

    fprintf(def_new, "%s", buffer1);
  }

  fclose(def_std);


  fclose(def_new);


  cout << "(setup): definitions.hpp is set up \n\n";


  /*_____________________________________________________________________________________________*/





  cout << "(setup): done, 3D-RT can now be compiled \n\n";

  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
