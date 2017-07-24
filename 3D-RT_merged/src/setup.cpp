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


  cout << "\n Setup for 3D-RT \n\n";

  cout << "Reading the parameters.txt file \n";



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




  /* Get all parameters for the line data files */





  int i,j;                                                                      /* level indices */

  int par1, par2, par3;                                         /* index for a collision partner */

  int lspec;                                    /* index of the line species under consideration */


  /* Get the number of levels and cumulatives for each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    nlev[lspec] = get_nlev(line_datafile[lspec]);

    cum_nlev[lspec] = 0;

    cum_nlev2[lspec] = 0;

    printf("(read_linedata): number of energy levels %d\n", nlev[lspec]);
  }


  /* Get the number of radiative transitions and cumulatives for each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    nrad[lspec] = get_nrad(line_datafile[lspec]);

    cum_nrad[lspec] = 0;
  }


  /* Calculate the cumulatives for nlev and nrad (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_nlev[lspec] = cum_nlev[lspec-1] + nlev[lspec-1];

    cum_nrad[lspec] = cum_nrad[lspec-1] + nrad[lspec-1];

    cum_nlev2[lspec] = cum_nlev2[lspec-1] + nlev[lspec-1]*nlev[lspec-1];
  }

  int tot_nlev = cum_nlev[NLSPEC-1] + nlev[NLSPEC-1];                      /* tot. nr. of levels */

  int tot_nrad = cum_nrad[NLSPEC-1] + nrad[NLSPEC-1];                 /* tot. nr. of transitions */

  int tot_nlev2 = cum_nlev2[NLSPEC-1] + nlev[NLSPEC-1]*nlev[NLSPEC-1];
                                                               /* tot of squares of nr of levels */



  /* Get the number of collision partners for each species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    ncolpar[lspec] = get_ncolpar(line_datafile[lspec]);

    cum_ncolpar[lspec] = 0;
  }


  /* Calculate the cumulative for ncolpar (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_ncolpar[lspec] = cum_ncolpar[lspec-1] + ncolpar[lspec-1];
  }

  int tot_ncolpar = cum_ncolpar[NLSPEC-1] + ncolpar[NLSPEC-1];



  /* Initialize the allocated memory */

  for (lspec=0; lspec<NLSPEC; lspec++){

    for (par1=0; par1<ncolpar[lspec]; par1++){

      ncoltran[LSPECPAR(lspec,par1)] = 0;
      cum_ncoltran[LSPECPAR(lspec,par1)] = 0;

      ncoltemp[LSPECPAR(lspec,par1)] = 0;
      cum_ncoltemp[LSPECPAR(lspec,par1)] = 0;

      cum_ncoltrantemp[LSPECPAR(lspec,par1)] = 0;
    }
  }


  /* For each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){


    /* For each collision partner */

    for (par2=0; par2<ncolpar[lspec]; par2++){


      /* Get the number of collisional transitions */

      ncoltran[LSPECPAR(lspec,par2)] = get_ncoltran(line_datafile[lspec], ncoltran, lspec);
/*
      printf( "(read_linedata): number of collisional transitions for partner %d is %d\n",
              par2, ncoltran[LSPECPAR(lspec,par2)] );
*/


      /* Get the number of collision temperatures */

      ncoltemp[LSPECPAR(lspec,par2)] = get_ncoltemp(line_datafile[lspec], ncoltran, par2, lspec);

/*
      printf( "(read_linedata): number of collisional temperatures for partner %d is %d\n",
              par2, ncoltemp[LSPECPAR(lspec,par2)] );
*/
    } /* end of par2 loop over collision partners */

  } /* end of lspec loop over line producing species */


  /* Calculate the cumulatives (needed for indexing, see definitions.h) */

  for (lspec=0; lspec<NLSPEC; lspec++){

    for (par3=1; par3<ncolpar[lspec]; par3++){

      cum_ncoltran[LSPECPAR(lspec,par3)] = cum_ncoltran[LSPECPAR(lspec,par3-1)]
                                             + ncoltran[LSPECPAR(lspec,par3-1)];

      cum_ncoltemp[LSPECPAR(lspec,par3)] = cum_ncoltemp[LSPECPAR(lspec,par3-1)]
                                             + ncoltemp[LSPECPAR(lspec,par3-1)];

      cum_ncoltrantemp[LSPECPAR(lspec,par3)] = cum_ncoltrantemp[LSPECPAR(lspec,par3-1)]
                                                 + ( ncoltran[LSPECPAR(lspec,par3-1)]
                                                     *ncoltemp[LSPECPAR(lspec,par3-1)] );
/*
      printf("(3D-RT): cum_ncoltran[%d] = %d \n", par3, cum_ncoltran[LSPECPAR(lspec,par3)]);
      printf("(3D-RT): cum_ncoltemp[%d] = %d \n", par3, cum_ncoltemp[LSPECPAR(lspec,par3)]);
      printf( "(3D-RT): cum_ncoltrantemp[%d] = %d \n",
              par3, cum_ncoltrantemp[LSPECPAR(lspec,par3)] );
*/
    }
  }


  for (lspec=0; lspec<NLSPEC; lspec++){

    tot_ncoltran[lspec] = cum_ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)]
                          + ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltemp[lspec] = cum_ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)]
                           + ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltrantemp[lspec] = cum_ncoltrantemp[LSPECPAR(lspec,ncolpar[lspec]-1)]
                              + ( ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)]
                                  *ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)] );
/*
    printf("(3D-RT): tot_ncoltran %d\n", tot_ncoltran[lspec]);
    printf("(3D-RT): tot_ncoltemp %d\n", tot_ncoltemp[lspec]);
    printf("(3D-RT): tot_ncoltrantemp %d\n", tot_ncoltrantemp[lspec]);
*/

    cum_tot_ncoltran[lspec] = 0;

    cum_tot_ncoltemp[lspec] = 0;

    cum_tot_ncoltrantemp[lspec] = 0;
  }


  /* Calculate the cumulatives of the cumulatives (also needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_tot_ncoltran[lspec] = cum_tot_ncoltran[lspec-1] + tot_ncoltran[lspec-1];

    cum_tot_ncoltemp[lspec] = cum_tot_ncoltemp[lspec-1] + tot_ncoltemp[lspec-1];

    cum_tot_ncoltrantemp[lspec] = cum_tot_ncoltrantemp[lspec-1] + tot_ncoltrantemp[lspec-1];
  }

  int tot_cum_tot_ncoltran = cum_tot_ncoltran[NLSPEC-1] + tot_ncoltran[NLSPEC-1];
                                                        /* total over the line prodcing species */
  int tot_cum_tot_ncoltemp = cum_tot_ncoltemp[NLSPEC-1] + tot_ncoltemp[NLSPEC-1];
                                                        /* total over the line prodcing species */
  int tot_cum_tot_ncoltrantemp = cum_tot_ncoltrantemp[NLSPEC-1]
                                   + tot_ncoltrantemp[NLSPEC-1];
                                                        /* total over the line prodcing species */







  cout << "grid file      : " << grid_inputfile << "\n";
  cout << "species file   : " << spec_datafile << "\n";
  cout << "line file      : " << line_datafile << "\n";
  cout << "reactions file : " << reac_datafile << "\n";
  cout << "ngrid          : " << ngrid << "\n";
  cout << "nsides         : " << nsides << "\n";
  cout << "nlspec         : " << nlspec << "\n";
  cout << "nspec          : " << nspec << "\n";

  cout << "\nSetting up definitions.hpp \n";











  /*   WRITE DEFINITIONS                                                                         */
  /*_____________________________________________________________________________________________*/



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


  cout << "\nSetup done, 3D-RT can now be compiled \n\n";

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
