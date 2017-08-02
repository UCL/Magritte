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




/* Grid related index definitions */

#define GINDEX(r,c) ( (c) + (r)*ngrid )                       /* when second index is grid point */

#define RINDEX(r,c) ( (c) + (r)*nrays )                              /* when second index is ray */

#define VINDEX(r,c) ( (c) + (r)*3 )                 /* when the second index is a 3-vector index */


/* Special key to find the number of the grid point associated to a certain evaluation point */

#define GP_NR_OF_EVALP(gridpoint, ray, evalp)                                                     \
        key[ GINDEX( (gridpoint), (evalp) + cum_raytot[RINDEX((gridpoint), (ray))] ) ]            \
               /* GP_NR_OF_EVALP(gridpoint, ray, evalp) gives the grid point number corresponding \
                       to the "evalp"'th evaluation point on ray "ray" of grid point "gridpoint" */


/* Level population related index definitions */

#define LSPECLEV(lspec,i)   ( (i) + cum_nlev[(lspec)] )                                           \
                               /* when first index is line producing species and second is level */

#define LSPECLEVLEV(lspec,i,j)   ( (j) + (i)*nlev[(lspec)] + cum_nlev2[(lspec)] )                 \
                   /* when first index is line producing species and second and third are levels */

#define LSPECGRIDLEVLEV(lspec,gridp,i,j)   ( (j) + (i)*nlev[(lspec)]                              \
                                             + (gridp)*nlev[(lspec)]*nlev[(lspec)]                \
                                             + ngrid*cum_nlev2[(lspec)] )                         \
/* when first index is line producing species, second is grid point, third and fourth are levels */

#define LSPECGRIDLEV(lspec,gridp,i)   ( (i) + (gridp)*nlev[(lspec)] + ngrid*cum_nlev[lspec] )     \
          /* when first index is line producing species, second is grid point and third is level */

#define LSPECRAD(lspec,kr)   ( (kr) + cum_nrad[(lspec)] )                                         \
                /* when first index is line producing species and second is radiative transition */

#define LSPECGRIDRAD(lspec,gridp,kr)   ( (kr) + (gridp)*nrad[(lspec)] + ngrid*cum_nlev[(lspec)] )
/* when first index is line producing species, second is grid point and third is rad. transition */


#define LINDEX(i,j) ((j)+(i)*nlev[lspec])                        /* when second index are levels */
#define TINDEX(r,c) ((c)+(r)*nrad[lspec])         /* when second index are radiative transitions */
#define L2INDEX(r,c) ((c)+(r)*nlev[lspec]*nlev[lspec])       /* when second index is LINDEX(i,j) */

// #define SINDEX(r,c) ((c)+(r)*nspec)               /* when second index are (chemical) species */

#define GRIDLEVLEV(g,i,j) (L2INDEX((g),LINDEX((i),(j)))) /* for a grid point and 2 level indices */


/* Collision rate related indices */

#define LSPECPAR(lspec,par)   ( (par) + cum_ncolpar[(lspec)] )                                    \
                   /* when first index is line producing species and second is collision partner */

#define LSPECPARTRAN(lspec,par,ctran)   ( (ctran) + cum_ncoltran[LSPECPAR((lspec),(par))]         \
				                                  + cum_tot_ncoltran[(lspec)] )                           \
                                  /* when first index line producing species, second is collision \
                                                     partner and third is collisional transition */

#define LSPECPARTEMP(lspec,par,ctemp)   ( (ctemp) + cum_ncoltemp[LSPECPAR((lspec),(par))]         \
                                          + cum_tot_ncoltemp[(lspec)] )		                        \
                               /* when first index is line producing species, second is collision \
                                                     partner and second is collision temperature */

#define LSPECPARTRANTEMP(lspec,par,ctran,ctemp)                                                   \
        ( (ctemp) + (ctran)*ncoltemp[LSPECPAR((lspec),(par))]                                     \
	        + cum_ncoltrantemp[LSPECPAR((lspec),(par))] + cum_tot_ncoltrantemp[(lspec)] )           \
                      /* when first index is line producing species, second is collision partner, \
                             third is collisional transition and fourth is collision temperature */





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



#define GRIDSPECRAY(gridp,spec,ray) (ray) + (spec)*nrays + (gridp)*nrays*nspec
               /* when the first index is a grid point, the second a species and the third a ray */





#endif /* __DECLARATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------*/
