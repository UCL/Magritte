/* Frederik De Ceuster - University College London & KU Leuven                                   */
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



#include "../parameters.hpp"
#include "Magritte_config.hpp"

#include <string>



// /*   Input before compilation, placed here by setup (src/setup.cpp)                              */
// /*_______________________________________________________________________________________________*/
//
//
// #define GRID_INPUTFILE "input/1Dn30.dat_conv.txt"
//
// #define SPEC_DATAFILE  "data/species_reduced.txt"
//
// #define REAC_DATAFILE  "data/rates_reduced.txt"
//
// #define LINE_DATAFILE0 "data/12C.txt"
//
// #define LINE_DATAFILE1 "data/12C+.txt"
//
// #define LINE_DATAFILE2 "data/16O.txt"
//
// #define LINE_DATAFILE3 "data/12CO.txt"
//
// #define NGRID 122
//
// #define NSIDES 1
//
// #define THETA_CRIT 1.300000
//
// #define RAY_SEPARATION2 0.000000E+00
//
// #define SOBOLEV true
//
// #define NSPEC 35
//
// #define NREAC 329
//
// #define NLSPEC 4
//
// #define TOT_NLEV 56
//
// #define TOT_NRAD 63
//
// #define TOT_NLEV2 1756
//
// #define TOT_NCOLPAR 18
//
// #define TOT_CUM_TOT_NCOLTRAN 1800
//
// #define TOT_CUM_TOT_NCOLTEMP 384
//
// #define TOT_CUM_TOT_NCOLTRANTEMP 44340
//
// #define MAX_NITERATIONS 300
//
// #define TIME_END_IN_YEARS 1.000000E+07
//
// #define FIELD_FORM "UNI"
//
// #define G_EXTERNAL_X 5.270460E+00
//
// #define G_EXTERNAL_Y 5.270460E+00
//
// #define G_EXTERNAL_Z 6.666670E+00
//
// #define IBC 0.0000000000000E+00
//
//
// /*_______________________________________________________________________________________________*/


#define ZETA   3.846153846153846
#define OMEGA  0.42


/* Numerical constants */

#define PI    3.141592653589793238462643383279502884197                                    /* pi */
#define CC    2.99792458E+10                                      /* speed of light in cgs units */
#define HH    6.62606896E-27                                   /* Planck's constant in cgs units */
#define KB    1.38065040E-16                                /* Boltzmann's constant in cgs units */
#define EV    1.60217646E-12                                             /* electron Volt in erg */
#define MP    1.67262164E-24                                         /* proton mass in cgs units */
#define PC    3.08568025E+18                                                   /* cm in a parsec */
#define AU    1.66053878E-24                                                 /* atomic mass unit */
#define T_CMB 2.725             /* temperature of the cosmic microwave background radiation in K */



/* Helper constants */

#define TOL 1.0E-9                                               /* tolerance for antipodal rays */
#define MAX_WIDTH 13                                                             /* for printing */
#define BUFFER_SIZE 500                                    /* max number of characters in a line */



/* Parameters for level population iteration */

#define POP_PREC        1.0E-2                        /* precision used in convergence criterion */
#define POP_LOWER_LIMIT 1.0E-26                                    /* lowest non-zero population */
#define POP_UPPER_LIMIT 1.0E+15                                            /* highest population */


/* Parameters for thermal balance iteration */

#define THERMAL_PREC 5.0E-3                           /* precision used in convergence criterion */


/* Grid related index definitions */

#define GINDEX(r,c) ( (c) + (r)*NGRID )                       /* when second index is grid point */

#define RINDEX(r,c) ( (c) + (r)*NRAYS )                              /* when second index is ray */

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
                                             + NGRID*cum_nlev2[(lspec)] )                         \
/* when first index is line producing species, second is grid point, third and fourth are levels */

#define LSPECGRIDLEV(lspec,gridp,i)   ( (i) + (gridp)*nlev[(lspec)] + NGRID*cum_nlev[lspec] )     \
          /* when first index is line producing species, second is grid point and third is level */

#define LSPECRAD(lspec,kr)   ( (kr) + cum_nrad[(lspec)] )                                         \
                /* when first index is line producing species and second is radiative transition */

#define LSPECGRIDRAD(lspec,gridp,kr)   ( (kr) + (gridp)*nrad[(lspec)] + NGRID*cum_nlev[(lspec)] )
/* when first index is line producing species, second is grid point and third is rad. transition */


#define LINDEX(i,j) ((j)+(i)*nlev[lspec])                        /* when second index are levels */
#define TINDEX(r,c) ((c)+(r)*nrad[lspec])         /* when second index are radiative transitions */
#define L2INDEX(r,c) ((c)+(r)*nlev[lspec]*nlev[lspec])       /* when second index is LINDEX(i,j) */

// #define SINDEX(r,c) ((c)+(r)*NSPEC)               /* when second index are (chemical) species */

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



/* Data types */

typedef struct GRIDPOINTS {

  double x, y, z;                                     /* x, y and z coordinate of the grid point */
  double vx, vy, vz;             /* x, y and z component of the velocity field of the grid point */

  double density;                                                   /* density at the grid point */

} GRIDPOINT;



typedef struct EVALPOINTS {

  bool   onray;             /* is true when the gridpoint is on any ray thus an evaluation point */

  long   ray;                               /* number of the ray the evaluation point belongs to */
  long   nr;                                     /* number of the evaluation point along the ray */

  long   eqp;                                                         /* point on equivalent ray */

  double dZ;                                                 /* distance increment along the ray */
  double Z;                                      /* distance between evaluation point and origin */
  double vol;                  /* velocity along the ray between grid point and evaluation point */

} EVALPOINT;



typedef struct SPECIES {

  std::string sym;                                                            /* chemical symbol */

  double mass;                                                                       /* mol mass */

  double abn[NGRID];                                                                /* abundance */

} SPECIES;



typedef struct REACTION {

  std::string   R1;                                                                /* reactant 1 */
  std::string   R2;                                                                /* reactant 2 */
  std::string   R3;                                                                /* reactant 3 */

  std::string   P1;                                                        /* reaction product 1 */
  std::string   P2;                                                        /* reaction product 2 */
  std::string   P3;                                                        /* reaction product 3 */
  std::string   P4;                                                        /* reaction product 4 */


  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */


  double k[NGRID];                                                  /* reaction rate coefficient */

  int    dup;                                           /* Number of duplicates of this reaction */

} REACTION;




/* Declaration of external variables */


/* Grid and evaluation points */

extern long cum_raytot[NGRID*NRAYS];    /* cumulative number of evaluation points along each ray */

extern long key[NGRID*NGRID];      /* stores the numbers of the grid points on the rays in order */

extern long raytot[NGRID*NRAYS];        /* cumulative number of evaluation points along each ray */


/* Level populations */

extern int nlev[NLSPEC];                                    /* number of levels for this species */

extern int nrad[NLSPEC];                     /* number of radiative transitions for this species */

extern int cum_nlev[NLSPEC];                         /* cumulative number of levels over species */

extern int cum_nlev2[NLSPEC];                    /* cumulative of squares of levels over species */

extern int cum_nrad[NLSPEC];       /* cumulative of number of radiative transitions over species */



extern int ncolpar[NLSPEC];                     /* number of collision partners for this species */

extern int cum_ncolpar[NLSPEC];          /* cumulative number of collision partners over species */

extern int ncoltemp[TOT_NCOLPAR];    /* number of col. temperatures for each species and partner */

extern int ncoltran[TOT_NCOLPAR];     /* number of col. transitions for each species and partner */

extern int cum_ncoltemp[TOT_NCOLPAR]; /* cum. nr. of col. temperatures over species and partners */

extern int cum_ncoltran[TOT_NCOLPAR];  /* cum. nr. of col. transitions over species and partners */

extern int tot_ncoltemp[NLSPEC];     /* total nr. of col. temperatures over species and partners */

extern int tot_ncoltran[NLSPEC];      /* total nr. of col. transitions over species and partners */

extern int cum_tot_ncoltemp[NLSPEC];     /* cum. of tot. of col. temp. over species and partners */

extern int cum_tot_ncoltran[NLSPEC]; /* cumulative tot. of col. trans. over species and partners */

extern int cum_ncoltrantemp[TOT_NCOLPAR]; /* cumulative of ntran*ntemp over species and partners */

extern int tot_ncoltrantemp[NLSPEC];           /* total of ntran*ntemp over species and partners */

extern int cum_tot_ncoltrantemp[NLSPEC];/* cum. of tot. of ntran*ntemp over species and partners */



/* Chemistry */

extern SPECIES species[NSPEC];

extern REACTION reaction[NREAC];


extern int lspec_nr[NLSPEC];                              /* names of the line producing species */

extern int spec_par[TOT_NCOLPAR];  /* number of the species corresponding to a collision partner */

extern char ortho_para[TOT_NCOLPAR];                    /* stores whether it is ortho or para H2 */



extern double metallicity;

extern double gas_to_dust;


#define GRIDSPECRAY(gridp,spec,ray)   ( (ray) + (spec)*NRAYS + (gridp)*NRAYS*NSPEC )
               /* when the first index is a grid point, the second a species and the third a ray */



/* Species numbers */

extern int e_nr;                                        /* species nr corresponding to electrons */

extern int H2_nr;                                              /* species nr corresponding to H2 */

extern int HD_nr;                                              /* species nr corresponding to HD */

extern int C_nr;                                                /* species nr corresponding to C */

extern int H_nr;                                                /* species nr corresponding to H */

extern int H2x_nr;                                            /* species nr corresponding to H2+ */

extern int HCOx_nr;                                          /* species nr corresponding to HCO+ */

extern int H3x_nr;                                            /* species nr corresponding to H3+ */

extern int H3Ox_nr;                                          /* species nr corresponding to H3O+ */

extern int Hex_nr;                                            /* species nr corresponding to He+ */

extern int CO_nr;                                              /* species nr corresponding to CO */


/* Reaction numbers */

extern int C_ionization_nr;

extern int H2_formation_nr;

extern int H2_photodissociation_nr;



extern double x_e;



#endif /* __DECLARATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
