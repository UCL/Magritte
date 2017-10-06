/* Frederik De Ceuster - University College London & KU Leuven                                   */
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

#include "declarations.hpp"
#include "setup_data_structures.hpp"
#include "data_tools.hpp"
#include "initializers.hpp"



/* setup_data_structures: set up all the different datastructures                                */
/*-----------------------------------------------------------------------------------------------*/

void setup_data_structures(string *line_datafile)
{

  setup_data_structures1(line_datafile);
  setup_data_structures2(line_datafile);

}

/*-----------------------------------------------------------------------------------------------*/





/* setup_data_structures1: set up the first part of the different datastructures                 */
/*-----------------------------------------------------------------------------------------------*/

void setup_data_structures1(string *line_datafile)
{


  int lspec;                                    /* index of the line species under consideration */



  for (lspec=0; lspec<NLSPEC; lspec++){


    /* Get the number of levels for each line producing species */

    nlev[lspec] = get_nlev(line_datafile[lspec]);


    /* Get the number of radiative transitions for each line producing species */

    nrad[lspec] = get_nrad(line_datafile[lspec]);

    // printf("(read_linedata): number of energy levels %d\n", nlev[lspec]);
    // printf("(read_linedata): number of radiative transitions %d\n", nrad[lspec]);
  }



  initialize_int_array(cum_nlev, NLSPEC);

  initialize_int_array(cum_nlev2, NLSPEC);

  initialize_int_array(cum_nrad, NLSPEC);



  /* Calculate the cumulatives for nlev and nrad (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_nlev[lspec] = cum_nlev[lspec-1] + nlev[lspec-1];

    cum_nrad[lspec] = cum_nrad[lspec-1] + nrad[lspec-1];

    cum_nlev2[lspec] = cum_nlev2[lspec-1] + nlev[lspec-1]*nlev[lspec-1];
  }




   /* Get the number of collision partners for each species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    ncolpar[lspec] = get_ncolpar(line_datafile[lspec]);

    // printf("(read_linedata): number of collisional partners %d\n", ncolpar[lspec]);
  }



  initialize_int_array(cum_ncolpar, NLSPEC);



  /* Calculate the cumulative for ncolpar (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_ncolpar[lspec] = cum_ncolpar[lspec-1] + ncolpar[lspec-1];
  }


}

/*-----------------------------------------------------------------------------------------------*/





/* setup_data_structures2: set up the second part of the different datastructures                */
/*-----------------------------------------------------------------------------------------------*/

void setup_data_structures2(string *line_datafile)
{


  int par;                                                      /* index for a collision partner */
  int lspec;                                    /* index of the line species under consideration */



  initialize_int_array(ncoltran, TOT_NCOLPAR);

  initialize_int_array(ncoltemp, TOT_NCOLPAR);



  /* For each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){


    /* For each collision partner */

    for (par=0; par<ncolpar[lspec]; par++){


      /* Get the number of collisional transitions */

      ncoltran[LSPECPAR(lspec,par)] = get_ncoltran(line_datafile[lspec], ncoltran, lspec);

      // printf( "(read_linedata): number of collisional transitions for partner %d is %d\n",
      //         par, ncoltran[LSPECPAR(lspec,par)] );



      /* Get the number of collision temperatures */

      ncoltemp[LSPECPAR(lspec,par)] = get_ncoltemp(line_datafile[lspec], ncoltran, par, lspec);

      // printf( "(read_linedata): number of collisional temperatures for partner %d is %d\n",
      //         par, ncoltemp[LSPECPAR(lspec,par)] );

    } /* end of par loop over collision partners */

  } /* end of lspec loop over line producing species */



  initialize_int_array(cum_ncoltran, TOT_NCOLPAR);

  initialize_int_array(cum_ncoltemp, TOT_NCOLPAR);

  initialize_int_array(cum_ncoltrantemp, TOT_NCOLPAR);



  /* Calculate the cumulatives (needed for indexing, see definitions.h) */

  for (lspec=0; lspec<NLSPEC; lspec++){

    for (par=1; par<ncolpar[lspec]; par++){

      cum_ncoltran[LSPECPAR(lspec,par)] = cum_ncoltran[LSPECPAR(lspec,par-1)]
                                          + ncoltran[LSPECPAR(lspec,par-1)];

      cum_ncoltemp[LSPECPAR(lspec,par)] = cum_ncoltemp[LSPECPAR(lspec,par-1)]
                                          + ncoltemp[LSPECPAR(lspec,par-1)];

      cum_ncoltrantemp[LSPECPAR(lspec,par)] = cum_ncoltrantemp[LSPECPAR(lspec,par-1)]
                                              + ( ncoltran[LSPECPAR(lspec,par-1)]
                                                  *ncoltemp[LSPECPAR(lspec,par-1)] );

      // printf("(Magritte): cum_ncoltran[%d] = %d \n", par, cum_ncoltran[LSPECPAR(lspec,par)]);
      // printf("(Magritte): cum_ncoltemp[%d] = %d \n", par, cum_ncoltemp[LSPECPAR(lspec,par)]);
      // printf( "(Magritte): cum_ncoltrantemp[%d] = %d \n",
      //         par, cum_ncoltrantemp[LSPECPAR(lspec,par)] );
    }
  }


  initialize_int_array(tot_ncoltran, NLSPEC);

  initialize_int_array(tot_ncoltemp, NLSPEC);

  initialize_int_array(tot_ncoltrantemp, NLSPEC);


  for (lspec=0; lspec<NLSPEC; lspec++){

    tot_ncoltran[lspec] = cum_ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)]
                          + ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltemp[lspec] = cum_ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)]
                          + ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltrantemp[lspec] = cum_ncoltrantemp[LSPECPAR(lspec,ncolpar[lspec]-1)]
                              + ( ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)]
                                  *ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)] );

    // printf("(Magritte): tot_ncoltran %d\n", tot_ncoltran[lspec]);
    // printf("(Magritte): tot_ncoltemp %d\n", tot_ncoltemp[lspec]);
    // printf("(Magritte): tot_ncoltrantemp %d\n", tot_ncoltrantemp[lspec]);
  }



  initialize_int_array(cum_tot_ncoltran, NLSPEC);

  initialize_int_array(cum_tot_ncoltemp, NLSPEC);

  initialize_int_array(cum_tot_ncoltrantemp, NLSPEC);



  /* Calculate the cumulatives of the cumulatives (also needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_tot_ncoltran[lspec] = cum_tot_ncoltran[lspec-1] + tot_ncoltran[lspec-1];

    cum_tot_ncoltemp[lspec] = cum_tot_ncoltemp[lspec-1] + tot_ncoltemp[lspec-1];

    cum_tot_ncoltrantemp[lspec] = cum_tot_ncoltrantemp[lspec-1] + tot_ncoltrantemp[lspec-1];
  }

}

/*-----------------------------------------------------------------------------------------------*/
