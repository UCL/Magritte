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

#include <string>
#include <iostream>
using namespace std;

#include "definitions.hpp"
#include "setup_data_structures.cpp"
#include "data_tools.cpp"



#define BUFFER_SIZE 500



/* main: Sets up the definitions.hpp file                                                        */
/*-----------------------------------------------------------------------------------------------*/

int main(){



  /*   READ PARAMETERS                                                                           */
  /*_____________________________________________________________________________________________*/


  string get_file(int nr);

  int get_nr(int nr);

  int get_nlev(string);

  int get_nrad(string);

  int get_ncolpar(string);

  int get_ncoltran(string, int*, int);

  int get_ncoltemp(string, int*, int, int);


  cout << "                \n";
  cout << "Setup for 3D-RT \n";
  cout << "--------------- \n\n";

  cout << "(setup): reading the parameters.txt file \n";



  /* Get nsides from parameters.txt */

  long nsides = get_nr(20);



  /* Get nlspec from parameters.txt */

  long nlspec = get_nr(21);



  /* Get the grid input file from line 10 in parameters.txt */

  string grid_inputfile = get_file(10);



  /* Get the number of grid points in input file */

  long get_NGRID(string grid_inputfile);                              /* defined in read_input.c */

  long ngrid = get_NGRID(grid_inputfile);             /* number of grid points in the input file */



  /* Get the species data file from line 11 in parameters.txt */

  string spec_datafile = get_file(11);



  /* Get the number of species from the species data file */

  int get_NSPEC(string spec_datafile);

  int nspec = get_NSPEC(spec_datafile);



  /* Get the line data file from line 12 in parameters.txt */

  string line_datafile[nlspec];

  line_datafile[0] = get_file(12);



  /* Get the reaction data file from line 13 in parameters.txt */

  string reac_datafile = get_file(13);



  /* Get the number of reactions from the reaction data file */

  int get_NREAC(string reac_datafile);

  int nreac = get_NREAC(reac_datafile);


  cout << "(setup): parameters.txt file read \n\n";


  /*_____________________________________________________________________________________________*/





  /*   EXTRACT PARAMETERS FROM THE LINE DATA                                                     */
  /*_____________________________________________________________________________________________*/


  cout << "(setup): extracting parameters from line data \n";


  /* Setup data structures */

  void setup_data_structures();

  setup_data_structures();



  int tot_nlev = cum_nlev[NLSPEC-1] + nlev[NLSPEC-1];                      /* tot. nr. of levels */

  int tot_nrad = cum_nrad[NLSPEC-1] + nrad[NLSPEC-1];                 /* tot. nr. of transitions */

  int tot_nlev2 = cum_nlev2[NLSPEC-1] + nlev[NLSPEC-1]*nlev[NLSPEC-1];
                                                               /* tot of squares of nr of levels */


  int tot_ncolpar = cum_ncolpar[NLSPEC-1] + ncolpar[NLSPEC-1];


  int tot_cum_tot_ncoltran = cum_tot_ncoltran[NLSPEC-1] + tot_ncoltran[NLSPEC-1];
                                                         /* total over the line prodcing species */
  int tot_cum_tot_ncoltemp = cum_tot_ncoltemp[NLSPEC-1] + tot_ncoltemp[NLSPEC-1];
                                                         /* total over the line prodcing species */
  int tot_cum_tot_ncoltrantemp = cum_tot_ncoltrantemp[NLSPEC-1]
                                   + tot_ncoltrantemp[NLSPEC-1];
                                                         /* total over the line prodcing species */



  cout << "(setup): grid file      : " << grid_inputfile << "\n";
  cout << "(setup): species file   : " << spec_datafile << "\n";
  cout << "(setup): line file      : " << line_datafile[0] << "\n";
  cout << "(setup): reactions file : " << reac_datafile << "\n";
  cout << "(setup): ngrid          : " << ngrid << "\n";
  cout << "(setup): nsides         : " << nsides << "\n";
  cout << "(setup): nlspec         : " << nlspec << "\n";
  cout << "(setup): nspec          : " << nspec << "\n";

  cout << "(setup): parameters from line data extracted \n\n";


  /*_____________________________________________________________________________________________*/





  /*   WRITE DEFINITIONS                                                                         */
  /*_____________________________________________________________________________________________*/


  cout << "(setup): setting up definitions.hpp \n";


  char buffer1[BUFFER_SIZE];
  char buffer2[BUFFER_SIZE];



  FILE *def_new = fopen("src/definitions.hpp", "w");



  /* Write the header */

  FILE *def_head = fopen("src/definitions_hd.txt", "r");


  while ( !feof(def_head) ){

    fgets(buffer1, BUFFER_SIZE, def_head);

    fprintf(def_new, "%s", buffer1);
  }

  fclose(def_head);



  /* write the new definitions */

  fprintf( def_new, "#define GRID_INPUTFILE \"%s\" \n\n", grid_inputfile.c_str() );

  fprintf( def_new, "#define SPEC_DATAFILE  \"%s\" \n\n", spec_datafile.c_str() );

  fprintf( def_new, "#define REAC_DATAFILE  \"%s\" \n\n", reac_datafile.c_str() );

  fprintf( def_new, "#define LINE_DATAFILE  \"%s\" \n\n", line_datafile[0].c_str() );


  fprintf( def_new, "#define NGRID %ld \n\n", ngrid );

  fprintf( def_new, "#define NSIDES %ld \n\n", nsides );

  fprintf( def_new, "#define NSPEC %d \n\n", nspec );

  fprintf( def_new, "#define NREAC %d \n\n", nreac );

  fprintf( def_new, "#define NLSPEC %ld \n\n", nlspec );

  fprintf( def_new, "#define TOT_NLEV %d \n\n", tot_nlev );

  fprintf( def_new, "#define TOT_NRAD %d \n\n", tot_nrad );

  fprintf( def_new, "#define TOT_NLEV2 %d \n\n", tot_nlev2 );

  fprintf( def_new, "#define TOT_NCOLPAR %d \n\n", tot_ncolpar );

  fprintf( def_new, "#define TOT_CUM_TOT_NCOLTRAN %d \n\n", tot_cum_tot_ncoltran );

  fprintf( def_new, "#define TOT_CUM_TOT_NCOLTEMP %d \n\n", tot_cum_tot_ncoltemp );

  fprintf( def_new, "#define TOT_CUM_TOT_NCOLTRANTEMP %d \n\n", tot_cum_tot_ncoltrantemp );




  /* Write the standard part of definitions */

  FILE *def_std = fopen("src/definitions_std.txt", "r");


  while ( !feof(def_std) ){

    fgets(buffer1, BUFFER_SIZE, def_std);

    fprintf(def_new, "%s", buffer1);
  }

  fclose(def_std);


  fclose(def_new);


  cout << "(setup): definitions.hpp are set up \n\n";


  /*_____________________________________________________________________________________________*/





  cout << "(setup): done, 3D-RT can now be compiled \n\n";

  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* get_file: get the input file name from parameters.txt                                         */
/*-----------------------------------------------------------------------------------------------*/

string get_file(int line)
{

  char buffer1[BUFFER_SIZE];
  char buffer2[BUFFER_SIZE];


  /* Open the parameters.txt file */

  FILE *params = fopen("parameters.txt", "r");


  /* Skip the lines before the file name */

  for (int l=0; l<line-1; l++){

    fgets(buffer1, BUFFER_SIZE, params);
  }

  fgets(buffer1, BUFFER_SIZE, params);

  sscanf(buffer1, "%s %*[^\n]\n", buffer2);

  string filename = buffer2;


  fclose(params);


  return filename;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_nr: get the input number from parameters.txt                                              */
/*-----------------------------------------------------------------------------------------------*/

long get_nr(int line)
{

  char buffer1[BUFFER_SIZE];
  long buffer2;


  /* Open the parameters.txt file */

  FILE *params = fopen("parameters.txt", "r");


  /* Skip the lines before the file name */

  for (int l=0; l<line-1; l++){

    fgets(buffer1, BUFFER_SIZE, params);
  }

  fgets(buffer1, BUFFER_SIZE, params);

  sscanf(buffer1, "%ld %*[^\n]\n", &buffer2);

  long nr = buffer2;


  fclose(params);


  return nr;

}

/*-----------------------------------------------------------------------------------------------*/
