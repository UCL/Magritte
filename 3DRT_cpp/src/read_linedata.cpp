/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* read_linedata: Read the data files containing the line information                            */
/*                                                                                               */
/* (based on read_input in 3D-PDR)                                                               */
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





/* extract_spec_par: extract the species corresponding to the collision partner                  */
/*-----------------------------------------------------------------------------------------------*/

void extract_spec_par(char *buffer, int lspec, int par)
{

  int n;                                                                                /* index */

  char buffer2[BUFFER_SIZE];                                 /* possibly modified copy of buffer */

  int cursor, cursor2;                                          /* index of position in a string */

  char string1[10], string2[10], string3[10];          /* buffers for the symbols of the species */



  /* Addapt for inconsistencies in specification of collision partners */

    cursor2=0;
    buffer2[cursor2] = buffer[0];
    cursor2++;

    for (cursor=1; cursor<BUFFER_SIZE/3; cursor++ ){

      if ( (buffer[cursor] == '-') && (buffer[cursor-1] != 'o') && (buffer[cursor-1] != 'p') ){

        buffer2[cursor2] = ' ';
        cursor2++;
        buffer2[cursor2] = '-';
        cursor2++;
        buffer2[cursor2] = ' ';
        cursor2++;
      }  

      else {
        buffer2[cursor2] = buffer[cursor];
        cursor2++;
      }
    }

    sscanf( buffer2, "%d %s %s %s %*[^\n] \n", &n, &string1, &string2, &string3 );


    /* Note: string3 contains the name of the collision partner */

    string name = string3;
    

    /* Use one of the species_tools to find the corresponding species nr */

    int get_species_nr(string name);

    spec_par[LSPECPAR(lspec,par)] = get_species_nr(name);


    char check_ortho_para(string name);

    ortho_para[LSPECPAR(lspec,par)] = check_ortho_para(name);

}


/*-----------------------------------------------------------------------------------------------*/
  




/* read_linedata: read data files containing the line information in LAMBDA/RADEX format         */
/*-----------------------------------------------------------------------------------------------*/
 
void read_linedata( string datafile, int *irad, int *jrad, double *energy, double *weight,
                    double *frequency, double *A_coeff, double *B_coeff, double *coltemp,
                    double *C_data, int *icol, int *jcol, int lspec )
{


  int l;                                                     /* index of a text line in the file */
  int par1, par2, par3, par4;                                   /* index for a collision partner */
  int tindex1, tindex2;                                                     /* temperature index */

  int n;                                                                         /* helper index */
  int i, j;                                                                     /* level indices */

  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */

  double buff1, buff2, buff3, buff4;                                 /* buffers to load the data */


  /* Open data file */

  FILE *data = fopen(datafile.c_str(), "r");


  /* Skip first 7 lines */

  for (l=0; l<7; l++){

    fscanf(data, "%*[^\n]\n");
  }


  /* Read energy levels */

  for (l=0; l<nlev[lspec]; l++){

    fscanf( data, "%d %lf %lf %*[^\n]\n", &n, &buff1, &buff2 );

    energy[LSPECLEV(lspec,l)] = buff1;
    weight[LSPECLEV(lspec,l)] = buff2; 

    // printf( "(read_linedata): level energy and weight are %f \t %.1f \n",
    //         energy[LSPECLEV(lspec,l)], weight[LSPECLEV(lspec,l)] );

  }


  /* Skip the next 3 lines */

  for (l=0; l<3; l++){

    fscanf(data, "%*[^\n]\n");
  }


  /* Read transitions and Einstein A coefficients */

  for (l=0; l<nrad[lspec]; l++){

    fgets(buffer, BUFFER_SIZE, data);
    sscanf( buffer, "%d %d %d %lE %lE %*[^\n] \n", &n, &i, &j, &buff1, &buff2 );

    irad[LSPECRAD(lspec,l)] = i-1;           /* shift levels down by 1 to have the usual indexing */
    jrad[LSPECRAD(lspec,l)] = j-1;           /* shift levels down by 1 to have the usual indexing */

    A_coeff[LSPECLEVLEV(lspec,i-1,j-1)] = buff1;

    frequency[LSPECLEVLEV(lspec,i-1,j-1)] = buff2;

    // printf( "(read_linedata): i, j, A_ij and frequency are %d \t %d \t %lE \t %lE \n",
    //         i-1, j-1, A_coeff[LSPECLEVLEV(lspec,irad[LSPECRAD(lspec,l)],jrad[LSPECRAD(lspec,l)])],
    //                 frequency[LSPECLEVLEV(lspec,irad[LSPECRAD(lspec,l)],jrad[LSPECRAD(lspec,l)])] );

  }



  /* Skip the next 2 lines */

  for (l=0; l<2; l++){

    fscanf(data, "%*[^\n]\n");
  }



  /* For each collision partner */

  for (par4=0; par4<ncolpar[lspec]; par4++){


    /* Skip the next line */

    fscanf(data, "%*[^\n]\n");


    /* Extract the species corresponding to the collision partner */

    fgets(buffer, BUFFER_SIZE, data);

  
    void extract_spec_par(char *buffer, int lspec, int par4);                   /* defined above */

    extract_spec_par(buffer, lspec, par4);
  

    /* Skip the next 5 lines */

    for (l=0; l<5; l++){

      fscanf(data, "%*[^\n]\n");
    }


    /* Read the collision temperatures */

    for (tindex1=0; tindex1<ncoltemp[LSPECPAR(lspec,par4)]; tindex1++){

      fscanf( data, "\t %lf \t", &buff3 );
      coltemp[LSPECPARTEMP(lspec,par4,tindex1)] = buff3;

      // printf( "(read_linedata): collisional temperature %*.2lf K\n", MAX_WIDTH,
      //         coltemp[LSPECPARTEMP(lspec,par4,tindex1)] );

    }


    /* Go to the next line (previous fscanf() did not do that!) */

    fscanf(data, "%*[^\n]\n");


    /* Read the collision rates */

    // printf("(read_linedata): C_data\n");

    for (l=0; l<ncoltran[LSPECPAR(lspec,par4)]; l++){


      /* Read first 3 entries of the line containing the transition level indices */

      fscanf( data, "%d \t %d \t %d \t", &n, &i, &j );

      icol[LSPECPARTRAN(lspec,par4,l)] = i-1; /* shift levels down by 1 to have the usual indexing */
      jcol[LSPECPARTRAN(lspec,par4,l)] = j-1; /* shift levels down by 1 to have the usual indexing */

      // printf( "\t i = %d   j = %d \n",
      //         icol[LSPECPARTRAN(lspec,par4,l)], jcol[LSPECPARTRAN(lspec,par4,l)] );



      /* Read the rest of the line containing the C_data */

      for (tindex2=0; tindex2<ncoltemp[LSPECPAR(lspec,par4)]; tindex2++){

        fscanf( data, "%lf", &buff4 );
        C_data[LSPECPARTRANTEMP(lspec,par4,l,tindex2)] = buff4;

        // printf("  %.2lE", C_data[LSPECPARTRANTEMP(lspec,par4,l,tindex2)]);

      }


      /* Go to the next line (previous fscanf() did not do that!) */

      fgets(buffer, BUFFER_SIZE, data);

      // printf("\n");

    }

    // printf("\n");


  } /* end of par4 loop over collision partners */


  fclose(data);



  /* Use data to calculate all coefficients in proper units */

  for (l=0; l<nrad[lspec]; l++){

    i = irad[LSPECRAD(lspec,l)];
    j = jrad[LSPECRAD(lspec,l)];


    /* Frequency is in GHz, convert to Hz */

    frequency[LSPECLEVLEV(lspec,i,j)] = 1.0E9*frequency[LSPECLEVLEV(lspec,i,j)];


    /* Energy/frequency of the transition is symmetric */

    frequency[LSPECLEVLEV(lspec,j,i)] = frequency[LSPECLEVLEV(lspec,i,j)];


    /* Calculate the Einstein B coefficients */

    B_coeff[LSPECLEVLEV(lspec,i,j)] = A_coeff[LSPECLEVLEV(lspec,i,j)] * pow(CC, 2)
                                     / ( 2.0*HH*pow(frequency[LSPECLEVLEV(lspec,i,j)] , 3) );

    B_coeff[LSPECLEVLEV(lspec,j,i)] = weight[LSPECLEV(lspec,i)] / weight[LSPECLEV(lspec,j)]
                                     * B_coeff[LSPECLEVLEV(lspec,i,j)];

    // printf( "(read_linedata): A_ij, B_ij and B_ji are  %lE \t %lE \t %lE \n",
    //         A_coeff[LSPECLEVLEV(lspec,i,j)], B_coeff[LSPECLEVLEV(lspec,i,j)],
    //         B_coeff[LSPECLEVLEV(lspec,j,i)] );

  }

  // printf("(read_linedata): intput C_data = \n");

  // for (int par=0; par<ncolpar[lspec]; par++){

  //   for (int ctran=0; ctran<ncoltran[LSPECPAR(lspec,par)]; ctran++){
  
  //     for (int ctemp=0; ctemp<ncoltemp[LSPECPAR(lspec,par)]; ctemp++){

  //       printf( "  %.2lE", C_data[LSPECPARTRANTEMP(lspec,par, ctran, ctemp)] );
  //     }
  
  //     printf("\n");
  //   }

  //   printf("\n");
  // }


}

/*-----------------------------------------------------------------------------------------------*/
