// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <iostream>

#include "setup.hpp"
#include "directories.hpp"
#include "setup_definitions.hpp"
#include "setup_data_structures.hpp"
#include "setup_data_tools.hpp"
#include "setup_initializers.hpp"
#include "setup_healpixvectors.hpp"


// main: Sets up Magritte_config.hpp file
// ------------- ------------------------

int main()
{


  printf("              \n");
  printf("Setup Magritte\n");
  printf("--------------\n");
  printf("              \n");




  // READ PARAMETERS
  // _______________


  printf("(setup): reading the parameters \n");


  // Get nrays from line 11 in PARAMETERS_FILE

# if (DIMENSIONS == 3)

    long nrays = 12*NSIDES*NSIDES;

# else

    long nrays = NRAYS;

# endif


  // Absolute path to project folder

  const std::string project_folder = PROJECT_FOLDER;


  // Get number of grid points from input file

  const std::string inputfile_rel = INPUTFILE;                        // relative
  const std::string inputfile     = project_folder + inputfile_rel;   // absolute



# if (FIXED_NCELLS)

#   if   (INPUT_FORMAT == '.vtu')

      long ncells = get_NCELLS_vtu(inputfile);

#   elif (INPUT_FORMAT == '.txt')

      long ncells = get_NCELLS_txt(inputfile);

#   endif

# endif



  // Get number of species from the species data file

  std::string spec_datafile = SPEC_DATAFILE;

  spec_datafile = "../" + spec_datafile;

  int nspec = get_NSPEC(spec_datafile);


  // Get line data files

  std::string line_datafile[NLSPEC] = LINE_DATAFILES;

  for (int l = 0; l < NLSPEC; l++)
  {
    line_datafile[l] = "../" + line_datafile[l];
  }


  // Get number of reactions from reaction data file

  std::string reac_datafile = REAC_DATAFILE;

  reac_datafile = "../" + reac_datafile;

  int nreac = get_NREAC(reac_datafile);


  std::string sobolev;

  if (SOBOLEV)
  {
    sobolev = "true";
  }

  else
  {
    sobolev = "false";
  }


  std::string field_form = FIELD_FORM;


  printf("\n");
  printf("(setup): parameters are: \n");

  printf("(setup):   input file        : %s\n", inputfile.c_str());
  printf("(setup):   species file      : %s\n", spec_datafile.c_str());
  printf("(setup):   reactions file    : %s\n", reac_datafile.c_str());

  printf("(setup):   NLSPEC            : %d\n", NLSPEC);

  for (int l = 0; l < NLSPEC; l++)
  {
    printf("(setup):   line file %d       : %s\n", l, line_datafile[l].c_str());
  }


# if (FIXED_NCELLS)

    printf("(setup):   ncells            : %ld\n", ncells);

# endif


  printf("(setup):   nsides            : %d\n",  NSIDES);
  printf("(setup):   nspec             : %d\n",  nspec);
  printf("(setup):   nrays             : %ld\n", nrays);
  printf("(setup):   sobolev           : %s\n",  sobolev.c_str());
  printf("(setup):   field_form        : %s\n",  field_form.c_str());
  printf("(setup):   time_end_in_years : %le\n", TIME_END_IN_YEARS);
  printf("(setup):   G_external_x      : %le\n", G_EXTERNAL_X);
  printf("(setup):   G_external_y      : %le\n", G_EXTERNAL_Y);
  printf("(setup):   G_external_z      : %le\n", G_EXTERNAL_Z);


  printf("\n");

  printf("(setup): parameters read \n\n");




  // EXTRACT PARAMETERS FROM LINE DATA
  // _________________________________


  printf("(setup): extracting parameters from line data \n");


  // Setup data structures

  int nlev[NLSPEC];

  initialize_int_array (NLSPEC, nlev);

  int nrad[NLSPEC];

  initialize_int_array (NLSPEC, nrad);

  int cum_nlev[NLSPEC];

  initialize_int_array (NLSPEC, cum_nlev);

  int cum_nlev2[NLSPEC];

  initialize_int_array (NLSPEC, cum_nlev2);

  int cum_nrad[NLSPEC];

  initialize_int_array (NLSPEC, cum_nrad);

  int ncolpar[NLSPEC];

  initialize_int_array (NLSPEC, ncolpar);

  int cum_ncolpar[NLSPEC];

  initialize_int_array (NLSPEC, cum_ncolpar);

  int max_nlev = 0;
  int max_nrad = 0;


  setup_data_structures1( line_datafile, nlev, nrad, cum_nlev, cum_nrad,
                          cum_nlev2, ncolpar, cum_ncolpar, &max_nlev, &max_nrad);



  int tot_nlev  = cum_nlev[NLSPEC-1]  + nlev[NLSPEC-1];                    /* tot. nr. of levels */

  int tot_nrad  = cum_nrad[NLSPEC-1]  + nrad[NLSPEC-1];               /* tot. nr. of transitions */

  int tot_nlev2 = cum_nlev2[NLSPEC-1] + nlev[NLSPEC-1]*nlev[NLSPEC-1];
                                                               /* tot of squares of nr of levels */

  int tot_ncolpar = cum_ncolpar[NLSPEC-1] + ncolpar[NLSPEC-1];


  int *ncoltemp;
  ncoltemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array (tot_ncolpar, ncoltemp);

  int *ncoltran;
  ncoltran = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array (tot_ncolpar, ncoltran);

  int *cum_ncoltemp;
  cum_ncoltemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array (tot_ncolpar, cum_ncoltemp);

  int *cum_ncoltran;
  cum_ncoltran = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array (tot_ncolpar, cum_ncoltran);

  int *cum_ncoltrantemp;
  cum_ncoltrantemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  initialize_int_array (tot_ncolpar, cum_ncoltrantemp);

  int tot_ncoltemp[NLSPEC];

  initialize_int_array (NLSPEC, tot_ncoltemp);

  int tot_ncoltran[NLSPEC];

  initialize_int_array (NLSPEC, tot_ncoltran);

  int cum_tot_ncoltemp[NLSPEC];

  initialize_int_array (NLSPEC, cum_tot_ncoltemp);

  int cum_tot_ncoltran[NLSPEC];

  initialize_int_array (NLSPEC, cum_tot_ncoltran);

  int tot_ncoltrantemp[NLSPEC];

  initialize_int_array (NLSPEC, tot_ncoltrantemp);

  int cum_tot_ncoltrantemp[NLSPEC];

  initialize_int_array (NLSPEC, cum_tot_ncoltrantemp);


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




  // SETUP HEALPIX VECTORS AND FIND ANTIPODAL PAIRS
  // ______________________________________________


  printf("(setup): creating HEALPix vectors \n");


  // Create (unit) HEALPix vectors and find antipodal pairs

  double *healpixvector = new double[3*nrays];   // array of HEALPix vectors

  long *antipod = new long[nrays];               // gives antipodal ray for each ray

  long **aligned = new long*[nrays];             // numbers of rays on same side

  for (long i = 0; i < nrays; i++)
  {
    aligned[i] = new long[nrays/2];
  }


  for (long i = 0; i < nrays; i++)
  {
    for (long j = 0; j < nrays/2; j++)
    {
      aligned[i][j] = 0;
    }
  }

  long *n_aligned = new long[nrays];             // number of rays on a particular side

  long *mirror_xz = new long[nrays];   // number of rays on a particular side


  setup_healpixvectors (nrays, healpixvector, antipod, n_aligned, aligned, mirror_xz);


  printf ("(setup): HEALPix vectors created \n\n");




  // WRITE CONFIG FILE
  // _________________


  printf ("(setup): setting up Magritte_config.hpp \n");


  FILE *config_file = fopen("../src/Magritte_config.hpp", "w");


# if (FIXED_NCELLS)

    fprintf( config_file, "#define NCELLS %ld \n\n", ncells );

# else

    fprintf( config_file, "#define NCELLS ncells \n\n");

# endif


# if (DIMENSIONS == 3)

    fprintf( config_file, "#define NRAYS %ld \n\n", nrays );

# endif


  fprintf (config_file, "#define NSPEC %d \n\n", nspec);

  fprintf (config_file, "#define NREAC %d \n\n", nreac);

  fprintf (config_file, "#define TOT_NLEV %d \n\n", tot_nlev);

  fprintf (config_file, "#define TOT_NRAD %d \n\n", tot_nrad);

  fprintf (config_file, "#define MAX_NLEV %d \n\n", max_nlev);

  fprintf (config_file, "#define MAX_NRAD %d \n\n", max_nrad);

  fprintf (config_file, "#define TOT_NLEV2 %d \n\n", tot_nlev2);

  fprintf (config_file, "#define TOT_NCOLPAR %d \n\n", tot_ncolpar);

  fprintf (config_file, "#define TOT_CUM_TOT_NCOLTRAN %d \n\n", tot_cum_tot_ncoltran);

  fprintf (config_file, "#define TOT_CUM_TOT_NCOLTEMP %d \n\n", tot_cum_tot_ncoltemp);

  fprintf (config_file, "#define TOT_CUM_TOT_NCOLTRANTEMP %d \n\n", tot_cum_tot_ncoltrantemp);


  write_int_array (config_file, "NLEV", nlev, NLSPEC);

  write_int_array (config_file, "NRAD", nrad, NLSPEC);


  write_int_array (config_file, "CUM_NLEV", cum_nlev, NLSPEC);

  write_int_array (config_file, "CUM_NLEV2", cum_nlev2, NLSPEC);

  write_int_array (config_file, "CUM_NRAD", cum_nrad, NLSPEC);


  write_int_array (config_file, "NCOLPAR", ncolpar, NLSPEC);

  write_int_array (config_file, "CUM_NCOLPAR", cum_ncolpar, NLSPEC);


  write_int_array (config_file, "NCOLTEMP", ncoltemp, tot_ncolpar);

  write_int_array (config_file, "NCOLTRAN", ncoltran, tot_ncolpar);


  write_int_array (config_file, "CUM_NCOLTEMP", cum_ncoltemp, tot_ncolpar);

  write_int_array (config_file, "CUM_NCOLTRAN", cum_ncoltran, tot_ncolpar);

  write_int_array (config_file, "CUM_NCOLTRANTEMP", cum_ncoltrantemp, tot_ncolpar);


  write_int_array (config_file, "TOT_NCOLTEMP", tot_ncoltemp, NLSPEC);

  write_int_array (config_file, "TOT_NCOLTRAN", tot_ncoltran, NLSPEC);

  write_int_array (config_file, "TOT_NCOLTRANTEMP", tot_ncoltrantemp, NLSPEC);


  write_int_array (config_file, "CUM_TOT_NCOLTEMP", cum_tot_ncoltemp, NLSPEC);

  write_int_array (config_file, "CUM_TOT_NCOLTRAN", cum_tot_ncoltran, NLSPEC);

  write_int_array (config_file, "CUM_TOT_NCOLTRANTEMP", cum_tot_ncoltrantemp, NLSPEC);


  write_double_array (config_file, "HEALPIXVECTOR", healpixvector, 3*nrays);

  write_long_array (config_file, "ANTIPOD", antipod, nrays);

  // write_long_array (config_file, "N_ALIGNED", n_aligned, nrays);

  // write_long_matrix (config_file, "ALIGNED", aligned, nrays, nrays/2);

  write_long_array (config_file, "MIRROR", mirror_xz, nrays);


  fclose (config_file);


  printf ("(setup): Magritte_config.hpp is set up \n\n");


  printf ("(setup): done, Magritte can now be compiled \n\n");


  return (0);

}




// write_int_array: write an array of int to config file
// -----------------------------------------------------

int write_int_array (FILE *file, std::string NAME, int *array, long length)
{

  fprintf( file, "#define %s { %d", NAME.c_str(), array[0]);

  for (long i = 1; i < length; i++)
  {
    fprintf( file, ", %d", array[i] );
  }

  fprintf( file, " } \n\n");


  return (0);

}




// write_long_array: write an array of long to config file
// -------------------------------------------------------

int write_long_array (FILE *file, std::string NAME, long *array, long length)
{

  fprintf( file, "#define %s { %ld", NAME.c_str(), array[0]);

  for (long i = 1; i < length; i++)
  {
    fprintf( file, ", %ld", array[i] );
  }

  fprintf( file, " } \n\n");


  return (0);

}




// write_long_matrix: write a matrix of longs to config file
// -------------------------------------------------------

int write_long_matrix (FILE *file, std::string NAME, long **array, long nrows, long ncols)
{

  fprintf( file, "#define %s {   \\\n", NAME.c_str());

  for (long i = 0; i< nrows; i++)
  {
    fprintf( file, "{ %ld", array[i][0]);

    for (long j = 1; j < ncols; j++)
    {
      fprintf (file, ", %ld", array[i][j]);
    }

    fprintf (file, " },   \\\n");
  }

  fprintf (file, " } \n");

  return (0);

}




// write_double_array: write an array of int to config file
// --------------------------------------------------------

int write_double_array (FILE *file, std::string NAME, double *array, long length)
{

  fprintf( file, "#define %s { %lE", NAME.c_str(), array[0]);

  for (long i = 1; i < length; i++)
  {
    fprintf( file, ", %lE", array[i] );
  }

  fprintf( file, " } \n\n");


  return (0);

}
