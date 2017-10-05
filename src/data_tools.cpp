/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* data_tools: tools to extract information form the data files                                  */
/*                                                                                               */
/* (based on read_species and read_rates in 3D-PDR)                                              */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include <string>
using namespace std;

#include "declarations.hpp"
#include "data_tools.hpp"



/* get_NGRID: Count number of grid points in input file input/iNGRID.txt                         */
/*-----------------------------------------------------------------------------------------------*/

long get_NGRID(string grid_inputfile)
{

  long ngrid=0;                                                         /* number of grid points */


  FILE *input1 = fopen(grid_inputfile.c_str(), "r");

  while ( !feof(input1) ){

    int ch = fgetc(input1);

    if (ch == '\n'){

      ngrid++;
    }

  }

  fclose(input1);

  return ngrid;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_NSPEC: get the number of species in the data file                                         */
/*-----------------------------------------------------------------------------------------------*/

int get_NSPEC(string spec_datafile)
{

  int nspec = 0;                                                            /* number of species */


  /* Open species data file */

  FILE *specdata1 = fopen(spec_datafile.c_str(), "r");

  while ( !feof(specdata1) ){

    int ch = fgetc(specdata1);

    if (ch == '\n'){

      nspec++;
    }

  }

  fclose(specdata1);


  /* Add two places for the dummy when a species is not found */

  nspec = nspec + 2;


  return nspec;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_NREAC: get the number of chemical reactions in the data file                              */
/*-----------------------------------------------------------------------------------------------*/

int get_NREAC(string reac_datafile)
{

  int nreac=0;                                                              /* number of species */


  /* Open species data file */

  FILE *reacdata1 = fopen(reac_datafile.c_str(), "r");

  while ( !feof(reacdata1) && EOF ){

    int ch = fgetc(reacdata1);

    if (ch == '\n'){

      nreac++;
    }

  }

  fclose(reacdata1);

  return nreac;

}

/*-----------------------------------------------------------------------------------------------*/




/* get_nlev: get number of energy levels from data file in LAMBDA/RADEX format                   */
/*-----------------------------------------------------------------------------------------------*/

int get_nlev(string datafile)
{

  int l;                                                     /* index of a text line in the file */
  int nlev=0;                                                                /* number of levels */


  /* Open data file */

  FILE *data1 = fopen(datafile.c_str(), "r");


  /* Skip first 5 lines */

  for (l=0; l<5; l++){

    fscanf(data1, "%*[^\n]\n");
  }


  /* Read the number of energy levels */

  fscanf(data1, "%d \n", &nlev);


  fclose(data1);

  return nlev;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_nrad: get number of radiative transitions from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_nrad(string datafile)
{

  int l;                                                     /* index of a text line in the file */
  int nrad=0;                                                 /* number of radiative transitions */

  int nlev = get_nlev(datafile);                                             /* number of levels */


  /* Open data file */

  FILE *data2 = fopen(datafile.c_str(), "r");


  /* Skip first 8+nlev lines */

  for (l=0; l<8+nlev; l++){

    fscanf(data2, "%*[^\n]\n");
  }


  /* Read the number of radiative transitions */

  fscanf(data2, "%d \n", &nrad);


  fclose(data2);

  return nrad;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_ncolpar: get number of collision partners from data file in LAMBDA/RADEX format           */
/*-----------------------------------------------------------------------------------------------*/

int get_ncolpar(string datafile)
{

  int l;                                                     /* index of a text line in the file */
  int ncolpar=0;                                                 /* number of collision partners */

  int nlev = get_nlev(datafile);                                             /* number of levels */
  int nrad = get_nrad(datafile);                              /* number of radiative transitions */


  /* Open data file */

  FILE *data3 = fopen(datafile.c_str(), "r");


  /* Skip first 11+nlev+nrad lines */

  for (l=0; l<11+nlev+nrad; l++){

    fscanf(data3, "%*[^\n]\n");
  }


  /* Read the number of collision partners */

  fscanf(data3, "%d \n", &ncolpar);


  fclose(data3);

  return ncolpar;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_ncoltran: get number of collisional transitions from data file in LAMBDA/RADEX format     */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltran(string datafile, int *ncoltran, int lspec)
{

  int l;                                                     /* index of a text line in the file */
  int loc_ncoltran=0;                                            /* number of collision partners */
  int par;                                                      /* index for a collision partner */

/*  int nlev = get_nlev(datafile);                                           /* number of levels */
/*  int nrad = get_nrad(datafile);                            /* number of radiative transitions */
/*  int ncolpar = get_ncolpar(datafile);                         /* number of collision partners */


  /* Open data file */

  FILE *data4 = fopen(datafile.c_str(), "r");


  /* Skip first 15+nlev+nrad lines */

  for (l=0; l<15+nlev[lspec]+nrad[lspec]; l++){

    fscanf(data4, "%*[^\n]\n");
  }


  /* Skip the collision partners that are already done */

  for (par=0; par<ncolpar[lspec]; par++){

    if (ncoltran[LSPECPAR(lspec,par)] > 0){

      /* Skip next 9+ncoltran lines */

      for (l=0; l<9+ncoltran[LSPECPAR(lspec,par)]; l++){

        fscanf(data4, "%*[^\n]\n");
      }
    }
  }


  /* Read the number of collisional transitions */

  fscanf(data4, "%d \n", &loc_ncoltran);


  fclose(data4);

  return loc_ncoltran;

}

/*-----------------------------------------------------------------------------------------------*/





/* get_ncoltemp: get number of collisional temperatures from data file in LAMBDA/RADEX format    */
/*-----------------------------------------------------------------------------------------------*/

int get_ncoltemp(string datafile, int *ncoltran, int partner, int lspec)
{

  int l;                                                     /* index of a text line in the file */
  int ncoltemp=0;                                                /* number of collision partners */
  int par;                                                      /* index for a collision partner */

  int nlev = get_nlev(datafile);                                             /* number of levels */
  int nrad = get_nrad(datafile);                              /* number of radiative transitions */
  int ncolpar = get_ncolpar(datafile);                           /* number of collision partners */


  /* Open data file */

  FILE *data5 = fopen(datafile.c_str(), "r");


  /* Skip first 17+nlev+nrad lines */

  for (l=0; l<17+nlev+nrad; l++){

    fscanf(data5, "%*[^\n]\n");
  }


  /* Skip the collision partners before "partner" */

  for (par=0; par<partner; par++){


    /* Skip next 9+ncoltran lines */

    for (l=0; l<9+ncoltran[LSPECPAR(lspec,par)]; l++){

      fscanf(data5, "%*[^\n]\n");
    }

  }


  /* Read the number of collisional temperatures */

  fscanf(data5, "%d \n", &ncoltemp);


  fclose(data5);

  return ncoltemp;

}

/*-----------------------------------------------------------------------------------------------*/
