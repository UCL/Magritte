// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <iostream>

#include "setup_definitions.hpp"
#include "setup_data_structures.hpp"
#include "setup_data_tools.hpp"



// setup_data_structures1: set up first part of different datastructures
// ---------------------------------------------------------------------

int setup_data_structures1 (std::string *line_datafile, int *nlev, int *nrad, int *cum_nlev,
                            int *cum_nrad, int *cum_nlev2, int *ncolpar, int *cum_ncolpar,
                            int *max_nlev, int *max_nrad)
{

  // For each line producing species

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {

    // Get number of levels

    nlev[lspec] = get_nlev (line_datafile[lspec]);

    if (nlev[lspec] > *max_nlev)
    {
      *max_nlev = nlev[lspec];
    }

    // Get number of radiative transitions

    nrad[lspec] = get_nrad (line_datafile[lspec]);

    if (nrad[lspec] > *max_nrad)
    {
      *max_nrad = nrad[lspec];
    }

    // printf("(read_linedata): number of energy levels %d\n", nlev[lspec]);
    // printf("(read_linedata): number of radiative transitions %d\n", nrad[lspec]);
  }


  // Calculate the cumulatives for nlev and nrad (needed for indexing, see definitions.hpp)

  for (int lspec = 1; lspec < NLSPEC; lspec++)
  {
    cum_nlev[lspec]  = cum_nlev[lspec-1]  + nlev[lspec-1];
    cum_nrad[lspec]  = cum_nrad[lspec-1]  + nrad[lspec-1];
    cum_nlev2[lspec] = cum_nlev2[lspec-1] + nlev[lspec-1]*nlev[lspec-1];
  }


  // Get number of collision partners for each species

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {
    ncolpar[lspec] = get_ncolpar(line_datafile[lspec]);

    // printf("(read_linedata): number of collisional partners %d\n", ncolpar[lspec]);
  }



  // Calculate cumulative for ncolpar (needed for indexing, see definitions.hpp)

  for (int lspec = 1; lspec < NLSPEC; lspec++)
  {
    cum_ncolpar[lspec] = cum_ncolpar[lspec-1] + ncolpar[lspec-1];
  }


  return (0);

}




// setup_data_structures2: set up second part of different datastructures
// ----------------------------------------------------------------------

int setup_data_structures2 (std::string *line_datafile, int* ncolpar, int *cum_ncolpar,
                            int *ncoltran, int *ncoltemp,
                            int *cum_ncoltran, int *cum_ncoltemp, int *cum_ncoltrantemp,
                            int *tot_ncoltran, int *tot_ncoltemp, int *tot_ncoltrantemp,
                            int *cum_tot_ncoltran, int *cum_tot_ncoltemp,
                            int *cum_tot_ncoltrantemp)
{

  // For each line producing species

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {

    // For each collision partner

    for (int par = 0; par < ncolpar[lspec]; par++)
    {

      // Get number of collisional transitions

      ncoltran[LSPECPAR(lspec,par)] = get_ncoltran (line_datafile[lspec], ncoltran, ncolpar,
                                                    cum_ncolpar, lspec);

      // printf( "(read_linedata): number of collisional transitions for partner %d is %d\n",
      //         par, ncoltran[LSPECPAR(lspec,par)] );


      // Get number of collision temperatures

      ncoltemp[LSPECPAR(lspec,par)] = get_ncoltemp (line_datafile[lspec], ncoltran, cum_ncolpar,
                                                    par, lspec);

      // printf( "(read_linedata): number of collisional temperatures for partner %d is %d\n",
      //         par, ncoltemp[LSPECPAR(lspec,par)] );

    } // end of par loop over collision partners

  } // end of lspec loop over line producing species



  // Calculate cumulatives (needed for indexing, see definitions.hpp)

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {
    for (int par = 1; par < ncolpar[lspec]; par++)
    {
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



  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {
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


  // Calculate cumulatives of cumulatives (also needed for indexing, see definitions.hpp)

  for (int lspec = 1; lspec < NLSPEC; lspec++)
  {
    cum_tot_ncoltran[lspec] = cum_tot_ncoltran[lspec-1] + tot_ncoltran[lspec-1];

    cum_tot_ncoltemp[lspec] = cum_tot_ncoltemp[lspec-1] + tot_ncoltemp[lspec-1];

    cum_tot_ncoltrantemp[lspec] = cum_tot_ncoltrantemp[lspec-1] + tot_ncoltrantemp[lspec-1];
  }


  return (0);

}
