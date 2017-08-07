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



/*   Input before compilation, placed here by setup (src/setup.cpp)                              */
/*_______________________________________________________________________________________________*/


#define GRID_INPUTFILE "input/grid_1D_regular.txt" 

#define SPEC_DATAFILE  "data/species_reduced.d" 

#define REAC_DATAFILE  "data/rates_reduced.d" 

#define LINE_DATAFILE0  "data/12c.dat" 

#define NGRID 101 

#define NSIDES 4 

#define THETA_CRIT 1.100000 

#define RAY_SEPARATION2 0.000000 

#define SOBOLEV false 

#define FIELD_FORM "ISO" 

#define NSPEC 34 

#define NREAC 329 

#define NLSPEC 1 

#define TOT_NLEV 5 

#define TOT_NRAD 7 

#define TOT_NLEV2 25 

#define TOT_NCOLPAR 6 

#define TOT_CUM_TOT_NCOLTRAN 60 

#define TOT_CUM_TOT_NCOLTEMP 141 

#define TOT_CUM_TOT_NCOLTRANTEMP 1410 


/*_______________________________________________________________________________________________*/





/* Input and data files */

extern string grid_inputfile;                   /* path to input file containing the grid points */

extern string spec_datafile;                         /* path to data file containing the species */

extern string reac_datafile;                       /* path to data file containing the reactions */

extern string line_datafile[NLSPEC];               /* path to data file containing the line data */






/* Numerical constants */

#define PI 3.141592653589793238462643383279502884197                                       /* pi */
#define CC 2.99792458E+10                                         /* speed of light in cgs units */
#define HH 6.62606896E-27                                      /* Planck's constant in cgs units */
#define KB 1.38065040E-16                                   /* Boltzmann's constant in cgs units */
#define EV 1.60217646E-12                                                /* electron Volt in erg */



/* Helper constants */

#define TOL 1.0E-9                                               /* tolerance for antipodal rays */
#define MAX_WIDTH 13                                                             /* for printing */
#define BUFFER_SIZE 500                                    /* max number of characters in a line */

#define NRAYS 12*NSIDES*NSIDES                   /* number of HEALPix rays as defined in HEALPix */


/* Parameters for level population iteration */

#define MAX_NITERATIONS 1

#define POP_PREC 1.0E-5                               /* precision used in convergence criterion */


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






/* ----- ADDITIONS for the chemistry code -----                                                  */
/* --------------------------------------------------------------------------------------------- */

#define AU 1.66053878E-24                                                    /* atomic mass unit */


typedef struct SPECIES {

  string sym;                                                                 /* chemical symbol */

  double mass;                                                                       /* mol mass */

  double abn[NGRID];                                                                /* abundance */

} SPECIES;



typedef struct REACTIONS {

  string   R1;                                                                     /* reactant 1 */
  string   R2;                                                                     /* reactant 2 */
  string   R3;                                                                     /* reactant 3 */

  string   P1;                                                             /* reaction product 1 */
  string   P2;                                                             /* reaction product 2 */
  string   P3;                                                             /* reaction product 3 */
  string   P4;                                                             /* reaction product 4 */


  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */


  double k;                                                         /* reaction rate coefficient */

  int    dup;                                           /* Number of duplicates of this reaction */

} REACTIONS;



extern SPECIES species[NSPEC];

extern REACTIONS reaction[NREAC];



extern int spec_par[TOT_NCOLPAR];  /* number of the species corresponding to a collision partner */

extern char ortho_para[TOT_NCOLPAR];                    /* stores whether it is ortho or para H2 */



extern double metallicity;

extern double gas2dust;


#define GRIDSPECRAY(gridp,spec,ray) (ray) + (spec)*NRAYS + (gridp)*NRAYS*NSPEC
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


