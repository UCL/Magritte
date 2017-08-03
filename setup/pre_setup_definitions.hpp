/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Definitions                                                                                   */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __DEFINITIONS_HPP_INCLUDED__
#define __DEFINITIONS_HPP_INCLUDED__

#include <string>
using namespace std;



/* Definitions of external variables */

string line_datafile[NLSPEC];


int *nlev;                                                  /* number of levels for this species */

int *nrad;                                   /* number of radiative transitions for this species */


int *cum_nlev;                                       /* cumulative number of levels over species */

int *cum_nlev2;                                  /* cumulative of squares of levels over species */

int *cum_nrad;                     /* cumulative of number of radiative transitions over species */


int *ncolpar;                                   /* number of collision partners for this species */

int *cum_ncolpar;                        /* cumulative number of collision partners over species */

int *ncoltemp;                       /* number of col. temperatures for each species and partner */

int *ncoltran;                        /* number of col. transitions for each species and partner */

int *cum_ncoltemp;                    /* cum. nr. of col. temperatures over species and partners */

int *cum_ncoltran;                     /* cum. nr. of col. transitions over species and partners */

int *tot_ncoltemp;                   /* total nr. of col. temperatures over species and partners */

int *tot_ncoltran;                    /* total nr. of col. transitions over species and partners */

int *cum_tot_ncoltemp;                   /* cum. of tot. of col. temp. over species and partners */

int *cum_tot_ncoltran;               /* cumulative tot. of col. trans. over species and partners */

int *cum_ncoltrantemp;                    /* cumulative of ntran*ntemp over species and partners */

int *tot_ncoltrantemp;                         /* total of ntran*ntemp over species and partners */

int *cum_tot_ncoltrantemp;              /* cum. of tot. of ntran*ntemp over species and partners */



#endif /* __DEFINITIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
