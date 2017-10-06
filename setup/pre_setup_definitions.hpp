/* Frederik De Ceuster - University College London & KU Leuven                                   */
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


#ifndef __DECLARATIONS_HPP_INCLUDED__
#define __DECLARATIONS_HPP_INCLUDED__
#ifndef __DEFINITIONS_HPP_INCLUDED__
#define __DEFINITIONS_HPP_INCLUDED__

#include <string>
using namespace std;



/* Helper constant */

#define BUFFER_SIZE 500                                    /* max number of characters in a line */



/* For compatibility */

extern int tot_ncolpar;
int tot_ncolpar;

#define TOT_NCOLPAR tot_ncolpar


typedef struct EVALPOINTS {

  bool   onray;             /* is true when the gridpoint is on any ray thus an evaluation point */

  long   ray;                               /* number of the ray the evaluation point belongs to */
  long   nr;                                     /* number of the evaluation point along the ray */

  long   eqp;                                                         /* point on equivalent ray */

  double dZ;                                                 /* distance increment along the ray */
  double Z;                                      /* distance between evaluation point and origin */
  double vol;                  /* velocity along the ray between grid point and evaluation point */

} EVALPOINT;



/* Collision rate related indices */

#define LSPECPAR(lspec,par)   ( (par) + cum_ncolpar[(lspec)] )                                    \
                   /* when first index is line producing species and second is collision partner */



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
#endif /* __DECLARATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
