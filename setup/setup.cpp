/* Frederik De Ceuster - University College London & KU Leuven                                   */
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

#include "setup.hpp"
#include "output_directory.hpp"
#include "setup_definitions.hpp"
#include "setup_data_structures.hpp"
#include "setup_data_tools.hpp"
#include "setup_initializers.hpp"
#include "setup_healpixvectors.hpp"



/* main: Sets up the definitions.hpp file                                                        */
/*-----------------------------------------------------------------------------------------------*/

int main()
{


  printf("                \n");
  printf("Setup Magritte  \n");
  printf("--------------\n\n");





  /*   READ PARAMETERS                                                                           */
  /*_____________________________________________________________________________________________*/


  printf("(setup): reading the parameters \n");


  /* Get nrays from line 11 in PARAMETERS_FILE */

# if (DIMENSIONS == 3)

  long nrays = 12*NSIDES*NSIDES;

# else

  long nrays = NRAYS;

# endif


  /* Get the number of grid points from the input file */

  std::string inputfile = INPUTFILE;

  inputfile = "../" + inputfile;


# if (INPUT_FORMAT == '.vtu')

  long ncells = get_NCELLS_vtu(inputfile);

# elif (INPUT_FORMAT == '.txt')

  long ncells = get_NCELLS_txt(inputfile);

# endif

  /* Get the number of species from the species data file */

  std::string spec_datafile = SPEC_DATAFILE;

  spec_datafile = "../" + spec_datafile;

  int nspec = get_NSPEC(spec_datafile);


  /* Get the line data files */

  std::string line_datafile[NLSPEC] = LINE_DATAFILES;

  for (int l=0; l<NLSPEC; l++)
  {
    line_datafile[l] = "../" + line_datafile[l];
  }


  /* Get the number of reactions from the reaction data file */

  std::string reac_datafile = REAC_DATAFILE;

  reac_datafile = "../" + reac_datafile;

  int nreac = get_NREAC(reac_datafile);


  std::string sobolev;
  if (SOBOLEV) sobolev = "true";
  else         sobolev = "false";


  std::string field_form = FIELD_FORM;


  printf("\n");
  printf("(setup): parameters are: \n");

  printf("(setup):   input file        : %s\n", inputfile.c_str());
  printf("(setup):   species file      : %s\n", spec_datafile.c_str());
  printf("(setup):   reactions file    : %s\n", reac_datafile.c_str());

  printf("(setup):   NLSPEC            : %d\n", NLSPEC);

  for (int l=0; l<NLSPEC; l++)
  {
    printf("(setup):   line file %d       : %s\n", l, line_datafile[l].c_str());
  }

  printf("(setup):   ncells            : %ld\n", ncells);
  printf("(setup):   nsides            : %d\n",  NSIDES);
  printf("(setup):   nspec             : %d\n",  nspec);
  printf("(setup):   nrays             : %ld\n", nrays);
  printf("(setup):   theta_crit        : %le\n", THETA_CRIT);
  printf("(setup):   sobolev           : %s\n",  sobolev.c_str());
  printf("(setup):   field_form        : %s\n",  field_form.c_str());
  printf("(setup):   time_end_in_years : %le\n", TIME_END_IN_YEARS);
  printf("(setup):   G_external_x      : %le\n", G_EXTERNAL_X);
  printf("(setup):   G_external_y      : %le\n", G_EXTERNAL_Y);
  printf("(setup):   G_external_z      : %le\n", G_EXTERNAL_Z);


  printf("\n");

  printf("(setup): parameters read \n\n");


  /*_____________________________________________________________________________________________*/





  /*   EXTRACT PARAMETERS FROM THE LINE DATA                                                     */
  /*_____________________________________________________________________________________________*/


  printf("(setup): extracting parameters from line data \n");


  /* Setup data structures */

  int nlev[NLSPEC];

  initialize_int_array(nlev, NLSPEC);

  int nrad[NLSPEC];

  initialize_int_array(nrad, NLSPEC);

  int cum_nlev[NLSPEC];

  initialize_int_array(cum_nlev, NLSPEC);

  int cum_nlev2[NLSPEC];

  initialize_int_array(cum_nlev2, NLSPEC);

  int cum_nrad[NLSPEC];

  initialize_int_array(cum_nrad, NLSPEC);

  int ncolpar[NLSPEC];

  initialize_int_array(ncolpar, NLSPEC);

  int cum_ncolpar[NLSPEC];

  initialize_int_array(cum_ncolpar, NLSPEC);



  setup_data_structures1( line_datafile, nlev, nrad, cum_nlev, cum_nrad,
                          cum_nlev2, ncolpar, cum_ncolpar );



  int tot_nlev  = cum_nlev[NLSPEC-1]  + nlev[NLSPEC-1];                    /* tot. nr. of levels */

  int tot_nrad  = cum_nrad[NLSPEC-1]  + nrad[NLSPEC-1];               /* tot. nr. of transitions */

  int tot_nlev2 = cum_nlev2[NLSPEC-1] + nlev[NLSPEC-1]*nlev[NLSPEC-1];
                                                               /* tot of squares of nr of levels */

  int tot_ncolpar = cum_ncolpar[NLSPEC-1] + ncolpar[NLSPEC-1];


  int *ncoltemp;
  ncoltemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array(ncoltemp, tot_ncolpar);

  int *ncoltran;
  ncoltran = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array(ncoltran, tot_ncolpar);

  int *cum_ncoltemp;
  cum_ncoltemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array(cum_ncoltemp, tot_ncolpar);

  int *cum_ncoltran;
  cum_ncoltran = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array(cum_ncoltran, tot_ncolpar);

  int *cum_ncoltrantemp;
  cum_ncoltrantemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array(cum_ncoltrantemp, tot_ncolpar);

  int tot_ncoltemp[NLSPEC];

  initialize_int_array(tot_ncoltemp, NLSPEC);

  int tot_ncoltran[NLSPEC];

  initialize_int_array(tot_ncoltran, NLSPEC);

  int cum_tot_ncoltemp[NLSPEC];

  initialize_int_array(cum_tot_ncoltemp, NLSPEC);

  int cum_tot_ncoltran[NLSPEC];

  initialize_int_array(cum_tot_ncoltran, NLSPEC);

  int tot_ncoltrantemp[NLSPEC];

  initialize_int_array(tot_ncoltrantemp, NLSPEC);

  int cum_tot_ncoltrantemp[NLSPEC];

  initialize_int_array(cum_tot_ncoltrantemp, NLSPEC);



  setup_data_structures2( line_datafile, ncolpar, cum_ncolpar,
                          ncoltran, ncoltemp, cum_ncoltran, cum_ncoltemp, cum_ncoltrantemp,
                          tot_ncoltran, tot_ncoltemp, tot_ncoltrantemp,
                          cum_tot_ncoltran, cum_tot_ncoltemp, cum_tot_ncoltrantemp );



  int tot_cum_tot_ncoltran = cum_tot_ncoltran[NLSPEC-1] + tot_ncoltran[NLSPEC-1];
                                                         /* total over the line prodcing species */
  int tot_cum_tot_ncoltemp = cum_tot_ncoltemp[NLSPEC-1] + tot_ncoltemp[NLSPEC-1];
                                                         /* total over the line prodcing species */
  int tot_cum_tot_ncoltrantemp = cum_tot_ncoltrantemp[NLSPEC-1] + tot_ncoltrantemp[NLSPEC-1];
                                                         /* total over the line prodcing species */


  printf("(setup): parameters extracted from line data \n\n");


  /*_____________________________________________________________________________________________*/





  /*   SETUP HEALPIX VECTORS AND FIND ANTIPODAL PAIRS                                            */
  /*_____________________________________________________________________________________________*/


  printf("(setup): creating HEALPix vectors \n");


  /* Create the (unit) HEALPix vectors and find antipodal pairs */

  double *healpixvector;                    /* array of HEALPix vectors for each ipix pixel */
  healpixvector = (double*) malloc( 3*nrays*sizeof(double) );

  long *antipod;                                             /* gives antipodal ray for each ray */
  antipod = (long*) malloc( nrays*sizeof(long) );


  setup_healpixvectors(nrays, healpixvector, antipod);


  printf("(setup): HEALPix vectors created \n\n");


  /*_____________________________________________________________________________________________*/





  /*   WRITE CONFIG FILE                                                                         */
  /*_____________________________________________________________________________________________*/


  printf("(setup): setting up Magritte_config.hpp \n");


  FILE *config_file = fopen("../src/Magritte_config.hpp", "w");

# if (FIXED_NCELLS)

  fprintf( config_file, "#define NCELLS %ld \n\n", ncells );

# else

  fprintf( config_file, "#define NCELLS ncells \n\n");

# endif


# if (DIMENSIONS == 3)

  fprintf( config_file, "#define NRAYS %ld \n\n", nrays );

# endif


  fprintf( config_file, "#define NSPEC %d \n\n", nspec );

  fprintf( config_file, "#define NREAC %d \n\n", nreac );

  fprintf( config_file, "#define TOT_NLEV %d \n\n", tot_nlev );

  fprintf( config_file, "#define TOT_NRAD %d \n\n", tot_nrad );

  fprintf( config_file, "#define TOT_NLEV2 %d \n\n", tot_nlev2 );

  fprintf( config_file, "#define TOT_NCOLPAR %d \n\n", tot_ncolpar );

  fprintf( config_file, "#define TOT_CUM_TOT_NCOLTRAN %d \n\n", tot_cum_tot_ncoltran );

  fprintf( config_file, "#define TOT_CUM_TOT_NCOLTEMP %d \n\n", tot_cum_tot_ncoltemp );

  fprintf( config_file, "#define TOT_CUM_TOT_NCOLTRANTEMP %d \n\n", tot_cum_tot_ncoltrantemp );


  write_int_array(config_file, "NLEV", nlev, NLSPEC);

  write_int_array(config_file, "NRAD", nrad, NLSPEC);


  write_int_array(config_file, "CUM_NLEV", cum_nlev, NLSPEC);

  write_int_array(config_file, "CUM_NLEV2", cum_nlev2, NLSPEC);

  write_int_array(config_file, "CUM_NRAD", cum_nrad, NLSPEC);


  write_int_array(config_file, "NCOLPAR", ncolpar, NLSPEC);

  write_int_array(config_file, "CUM_NCOLPAR", cum_ncolpar, NLSPEC);


  write_int_array(config_file, "NCOLTEMP", ncoltemp, tot_ncolpar);

  write_int_array(config_file, "NCOLTRAN", ncoltran, tot_ncolpar);


  write_int_array(config_file, "CUM_NCOLTEMP", cum_ncoltemp, tot_ncolpar);

  write_int_array(config_file, "CUM_NCOLTRAN", cum_ncoltran, tot_ncolpar);

  write_int_array(config_file, "CUM_NCOLTRANTEMP", cum_ncoltrantemp, tot_ncolpar);


  write_int_array(config_file, "TOT_NCOLTEMP", tot_ncoltemp, NLSPEC);

  write_int_array(config_file, "TOT_NCOLTRAN", tot_ncoltran, NLSPEC);

  write_int_array(config_file, "TOT_NCOLTRANTEMP", tot_ncoltrantemp, NLSPEC);


  write_int_array(config_file, "CUM_TOT_NCOLTEMP", cum_tot_ncoltemp, NLSPEC);

  write_int_array(config_file, "CUM_TOT_NCOLTRAN", cum_tot_ncoltran, NLSPEC);

  write_int_array(config_file, "CUM_TOT_NCOLTRANTEMP", cum_tot_ncoltrantemp, NLSPEC);


  write_double_array(config_file, "HEALPIXVECTOR", healpixvector, 3*nrays);

  write_long_array(config_file, "ANTIPOD", antipod, nrays);


  fclose(config_file);


  printf("(setup): Magritte_config.hpp is set up \n\n");


  /*_____________________________________________________________________________________________*/




  printf("(setup): done, Magritte can now be compiled \n\n");

  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_int_array: write an array of int to the config file                                     */
/*-----------------------------------------------------------------------------------------------*/

int write_int_array(FILE *file, std::string NAME, int *array, long length)
{

  fprintf( file, "#define %s { %d", NAME.c_str(), array[0]);

  for (long i=1; i<length; i++)
  {
    fprintf( file, ", %d", array[i] );
  }

  fprintf( file, " } \n\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_long_array: write an array of long to the config file                                   */
/*-----------------------------------------------------------------------------------------------*/

int write_long_array(FILE *file, std::string NAME, long *array, long length)
{

  fprintf( file, "#define %s { %ld", NAME.c_str(), array[0]);

  for (long i=1; i<length; i++)
  {
    fprintf( file, ", %ld", array[i] );
  }

  fprintf( file, " } \n\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* write_double_array: write an array of int to the config file                                  */
/*-----------------------------------------------------------------------------------------------*/

int write_double_array(FILE *file, std::string NAME, double *array, long length)
{

  fprintf( file, "#define %s { %lE", NAME.c_str(), array[0]);

  for (long i=1; i<length; i++)
  {
    fprintf( file, ", %lE", array[i] );
  }

  fprintf( file, " } \n\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
