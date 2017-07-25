/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* data_structures: set up the different datastructures (managing indices, linearizing arrays)   */
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


/* setup_data_structures: set up the different datastructures                                    */
/*-----------------------------------------------------------------------------------------------*/

void setup_data_structures()
{

  int i,j;                                                                      /* level indices */

  int par1, par2, par3;                                         /* index for a collision partner */

  int lspec;                                    /* index of the line species under consideration */


  int get_nlev(string);
  
  int get_nrad(string);
  
  int get_ncolpar(string);
  
  int get_ncoltran(string, int*, int);
  
  int get_ncoltemp(string, int*, int, int);


  /* Read data files */

  line_datafile[0] = LINE_DATAFILE;
  // line_datafile[0] = "data/12c+.dat";
  // line_datafile[0] = "data/12co.dat";
  // line_datafile[0] = "data/16o.dat";


/* P1 */

 /* Get the number of levels and cumulatives for each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    nlev[lspec] = get_nlev(line_datafile[lspec]);

    cum_nlev[lspec] = 0;

    cum_nlev2[lspec] = 0;

    // printf("(read_linedata): number of energy levels %d\n", nlev[lspec]);
  }


  /* Get the number of radiative transitions and cumulatives for each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    nrad[lspec] = get_nrad(line_datafile[lspec]);

    cum_nrad[lspec] = 0;

    // printf("(read_linedata): number of radiative transitions %d\n", nrad[lspec]);
  }


  /* Calculate the cumulatives for nlev and nrad (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_nlev[lspec] = cum_nlev[lspec-1] + nlev[lspec-1];

    cum_nrad[lspec] = cum_nrad[lspec-1] + nrad[lspec-1];

    cum_nlev2[lspec] = cum_nlev2[lspec-1] + nlev[lspec-1]*nlev[lspec-1];
  }


  /* P2 */

   /* Get the number of collision partners for each species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    ncolpar[lspec] = get_ncolpar(line_datafile[lspec]);

    cum_ncolpar[lspec] = 0;

    // printf("(read_linedata): number of collisional partners %d\n", ncolpar[lspec]);
  }


  /* Calculate the cumulative for ncolpar (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_ncolpar[lspec] = cum_ncolpar[lspec-1] + ncolpar[lspec-1];
  }



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

}

/*-----------------------------------------------------------------------------------------------*/
