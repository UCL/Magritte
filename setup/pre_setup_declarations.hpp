/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Declarations                                                                                  */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __DECLARATIONS_HPP_INCLUDED__
#define __DECLARATIONS_HPP_INCLUDED__

#include <string>
using namespace std;

#include "NLSPEC.hpp"



/* Input and data files */

extern string grid_inputfile;                   /* path to input file containing the grid points */

extern string spec_datafile;                         /* path to data file containing the species */

extern string reac_datafile;                       /* path to data file containing the reactions */

extern string line_datafile[NLSPEC];               /* path to data file containing the line data */



/* Helper constant */

#define BUFFER_SIZE 500                                    /* max number of characters in a line */



/* Collision rate related indices */

#define LSPECPAR(lspec,par)   ( (par) + cum_ncolpar[(lspec)] )                                    \
                   /* when first index is line producing species and second is collision partner */



/* Declaration of external variables */

extern int *nlev;                                           /* number of levels for this species */

extern int *nrad;                            /* number of radiative transitions for this species */

extern int *cum_nlev;                                /* cumulative number of levels over species */

extern int *cum_nlev2;                           /* cumulative of squares of levels over species */

extern int *cum_nrad;              /* cumulative of number of radiative transitions over species */



extern int *ncolpar;                            /* number of collision partners for this species */

extern int *cum_ncolpar;                 /* cumulative number of collision partners over species */

extern int *ncoltemp;                /* number of col. temperatures for each species and partner */

extern int *ncoltran;                 /* number of col. transitions for each species and partner */

extern int *cum_ncoltemp;             /* cum. nr. of col. temperatures over species and partners */

extern int *cum_ncoltran;              /* cum. nr. of col. transitions over species and partners */

extern int *tot_ncoltemp;            /* total nr. of col. temperatures over species and partners */

extern int *tot_ncoltran;             /* total nr. of col. transitions over species and partners */

extern int *cum_tot_ncoltemp;            /* cum. of tot. of col. temp. over species and partners */

extern int *cum_tot_ncoltran;        /* cumulative tot. of col. trans. over species and partners */

extern int *cum_ncoltrantemp;             /* cumulative of ntran*ntemp over species and partners */

extern int *tot_ncoltrantemp;                  /* total of ntran*ntemp over species and partners */

extern int *cum_tot_ncoltrantemp;       /* cum. of tot. of ntran*ntemp over species and partners */



#endif /* __DECLARATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
