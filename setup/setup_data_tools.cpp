/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* setup_data_tools: tools to extract information form the data files                            */
/*                                                                                               */
/* (based on read_species and read_rates in 3D-PDR)                                              */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "setup_definitions.hpp"
#include "setup_data_tools.hpp"



/* get_NGRID: Count number of grid points in input file input/iNGRID.txt                         */
/*-----------------------------------------------------------------------------------------------*/

long get_NGRID(std::string grid_inputfile)
{


  long ngrid = 0;                                                       /* number of grid points */


  FILE *file = fopen(grid_inputfile.c_str(), "r");

  while ( !feof(file) ){

    int ch = fgetc(file);

    if (ch == '\n'){

      ngrid++;
    }

  }

  fclose(file);


  return ngrid;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_NSPEC: get the number of species in the data file                                         */
/*-----------------------------------------------------------------------------------------------*/

int get_NSPEC(std::string spec_datafile)
{


  int nspec = 0;                                                            /* number of species */


  /* Open species data file */

  FILE *file = fopen(spec_datafile.c_str(), "r");

  while ( !feof(file) ){

    int ch = fgetc(file);

    if (ch == '\n'){

      nspec++;
    }

  }

  fclose(file);


  /* Add two places, one for the dummy when a species is not found and one for the total */

  nspec = nspec + 2;


  return nspec;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_NREAC: get the number of chemical reactions in the data file                              */
/*-----------------------------------------------------------------------------------------------*/

int get_NREAC(std::string reac_datafile)
{


  int nreac = 0;                                                            /* number of species */


  /* Open species data file */

  FILE *file = fopen(reac_datafile.c_str(), "r");

  while ( !feof(file) && EOF ){

    int ch = fgetc(file);

    if (ch == '\n'){

      nreac++;
    }

  }

  fclose(file);


  return nreac;

}

/*-----------------------------------------------------------------------------------------------*/




/* get_nlev: get number of energy levels from data file in LAMBDA/RADEX format                   */
/*-----------------------------------------------------------------------------------------------*/

int get_nlev(std::string datafile)
{


  int nlev = 0;                                                              /* number of levels */


  /* Open data file */

  FILE *file = fopen(datafile.c_str(), "r");


  /* Skip first 5 lines */

  for (int l=0; l<5; l++){

    fscanf(file, "%*[^\n]\n");
  }


  /* Read the number of energy levels */

  fscanf(file, "%d \n", &nlev);


  fclose(file);


  return nlev;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_nrad: get number of radiative transitions from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_nrad(std::string datafile)
{


  int nrad = 0;                                               /* number of radiative transitions */

  int nlev = get_nlev(datafile);                                             /* number of levels */


  /* Open data file */

  FILE *file = fopen(datafile.c_str(), "r");


  /* Skip first 8+nlev lines */

  for (int l=0; l<8+nlev; l++){

    fscanf(file, "%*[^\n]\n");
  }


  /* Read the number of radiative transitions */

  fscanf(file, "%d \n", &nrad);


  fclose(file);


  return nrad;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_ncolpar: get number of collision partners from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_ncolpar(std::string datafile)
{


  int ncolpar = 0;                                               /* number of collision partners */

  int nlev = get_nlev(datafile);                                             /* number of levels */
  int nrad = get_nrad(datafile);                              /* number of radiative transitions */


  /* Open data file */

  FILE *file = fopen(datafile.c_str(), "r");


  /* Skip first 11+nlev+nrad lines */

  for (int l=0; l<11+nlev+nrad; l++){

    fscanf(file, "%*[^\n]\n");
  }


  /* Read the number of collision partners */

  fscanf(file, "%d \n", &ncolpar);


  fclose(file);


  return ncolpar;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_ncoltran: get number of collisional transitions from data file in LAMBDA/RADEX format     */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltran(std::string datafile, int *ncoltran, int *ncolpar, int *cum_ncolpar, int lspec)
{


  int loc_ncoltran=0;                                            /* number of collision partners */

  int nlev = get_nlev(datafile);                                             /* number of levels */
  int nrad = get_nrad(datafile);                              /* number of radiative transitions */


  /* Open data file */

  FILE *file = fopen(datafile.c_str(), "r");


  /* Skip first 15+nlev+nrad lines */

  for (int l=0; l<15+nlev+nrad; l++){

    fscanf(file, "%*[^\n]\n");
  }


  /* Skip the collision partners that are already done */

  for (int par=0; par<ncolpar[lspec]; par++){

    if (ncoltran[LSPECPAR(lspec,par)] > 0){

      /* Skip next 9+ncoltran lines */

      for (int l=0; l<9+ncoltran[LSPECPAR(lspec,par)]; l++){

        fscanf(file, "%*[^\n]\n");
      }
    }
  }


  /* Read the number of collisional transitions */

  fscanf(file, "%d \n", &loc_ncoltran);


  fclose(file);


  return loc_ncoltran;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_ncoltemp: get number of collisional temperatures from data file in LAMBDA/RADEX format    */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltemp(std::string datafile, int *ncoltran, int *cum_ncolpar, int partner, int lspec)
{


  int ncoltemp = 0;                                              /* number of collision partners */

  int nlev    = get_nlev(datafile);                                          /* number of levels */
  int nrad    = get_nrad(datafile);                           /* number of radiative transitions */
  int ncolpar = get_ncolpar(datafile);                           /* number of collision partners */


  /* Open data file */

  FILE *file = fopen(datafile.c_str(), "r");


  /* Skip first 17+nlev+nrad lines */

  for (int l=0; l<17+nlev+nrad; l++){

    fscanf(file, "%*[^\n]\n");
  }


  /* Skip the collision partners before "partner" */

  for (int par=0; par<partner; par++){


    /* Skip next 9+ncoltran lines */

    for (int l=0; l<9+ncoltran[LSPECPAR(lspec,par)]; l++){

      fscanf(file, "%*[^\n]\n");
    }

  }


  /* Read the number of collisional temperatures */

  fscanf(file, "%d \n", &ncoltemp);


  fclose(file);


  return ncoltemp;

}

/*-----------------------------------------------------------------------------------------------*/