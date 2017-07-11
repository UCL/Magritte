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





/* Numerical constants */

#define PI 3.141592653589793238462643383279502884197                                       /* pi */
#define CC 2.99792458E+10                                         /* speed of light in cgs units */
#define HH 6.62606896E-27                                      /* Planck's constant in cgs units */
#define KB 1.38065040E-16                                   /* Boltzmann's constant in cgs units */


/* Helper constants */

#define TOL 1.0E-9                                               /* tolerance for antipodal rays */
#define MAX_WIDTH 13                                                             /* for printing */
#define BUFFER_SIZE 500                                    /* max number of characters in a line */

#define NRAYS 12*nsides*nsides                   /* number of HEALPix rays as defined in HEALPix */


/* Parameters for level population iteration */

#define MAX_NITERATIONS 1

#define POP_PREC 1.0E-5                               /* precision used in convergence criterion */


/* Grid related index definitions */

#define GINDEX(r,c) ( (c) + (r)*ngrid )                       /* when second index is grid point */

#define RINDEX(r,c) ( (c) + (r)*NRAYS )                              /* when second index is ray */

#define VINDEX(r,c) ( (c) + (r)*3 )                 /* when the second index is a 3-vector index */


/* Special key to find the number of the grid point associated to a certain evaluation point */

#define GP_NR_OF_EVALP(gridpoint, ray, evalp)                                                     \
        key[ GINDEX( (gridpoint), (evalp) + cum_raytot[RINDEX((gridpoint), (ray))] ) ]            \
               /* GP_NR_OF_EVALP(gridpoint, ray, evalp) gives the grid point number corresponding \
                       to the "evalp"'th evaluation point on ray "ray" of grid point "gridpoint" */


/* Level population related index definitions */

#define SPECLEV(lspec,i)   ( (i) + cum_nlev[(lspec)] )                                            \
                               /* when first index is line producing species and second is level */

#define SPECLEVLEV(lspec,i,j)   ( (j) + (i)*nlev[(lspec)] + cum_nlev2[(lspec)] )                  \
                   /* when first index is line producing species and second and third are levels */

#define SPECGRIDLEVLEV(lspec,gridp,i,j)   ( (j) + (i)*nlev[(lspec)]                               \
                                             + (gridp)*nlev[(lspec)]*nlev[(lspec)]                \
                                             + ngrid*cum_nlev2[(lspec)] )                         \
/* when first index is line producing species, second is grid point, third and fourth are levels */

#define SPECGRIDLEV(lspec,gridp,i)   ( (i) + (gridp)*nlev[(lspec)] + ngrid*cum_nlev[lspec] )      \
          /* when first index is line producing species, second is grid point and third is level */

#define SPECRAD(lspec,kr)   ( (kr) + cum_nrad[(lspec)] )                                          \
                /* when first index is line producing species and second is radiative transition */


#define LINDEX(i,j) ((j)+(i)*nlev[lspec])                        /* when second index are levels */
#define TINDEX(r,c) ((c)+(r)*nrad[lspec])         /* when second index are radiative transitions */
#define L2INDEX(r,c) ((c)+(r)*nlev[lspec]*nlev[lspec])       /* when second index is LINDEX(i,j) */

#define SINDEX(r,c) ((c)+(r)*nspec)                  /* when second index are (chemical) species */

#define GRIDLEVLEV(g,i,j) (L2INDEX((g),LINDEX((i),(j)))) /* for a grid point and 2 level indices */


/* Collision rate related indices */

#define SPECPAR(lspec,par)   ( (par) + cum_ncolpar[(lspec)] )                                     \
                   /* when first index is line producing species and second is collision partner */

#define SPECPARTRAN(lspec,par,ctran)   ( (ctran) + cum_ncoltran[SPECPAR((lspec),(par))]      \
				          + cum_tot_ncoltran[(lspec)] )                           \
                                  /* when first index line producing species, second is collision \
                                                     partner and third is collisional transition */

#define SPECPARTEMP(lspec,par,ctemp)   ( (ctemp) + cum_ncoltemp[SPECPAR((lspec),(par))]      \
                                          + cum_tot_ncoltemp[(lspec)] )		                  \
                               /* when first index is line producing species, second is collision \
                                                     partner and second is collision temperature */

#define SPECPARTRANTEMP(lspec,par,ctran,ctemp)                                                    \
        ( (ctemp) + (ctran)*ncoltemp[SPECPAR((lspec),(par))]                                 \
	   + cum_ncoltrantemp[SPECPAR((lspec),(par))] + cum_tot_ncoltrantemp[(lspec)] )      \
                      /* when first index is line producing species, second is collision partner, \
                             third is collisional transition and fourth is collision temperature */



/* Data types */

// typedef struct GRIDPOINTS {

//   double x, y, z;                                     /* x, y and z coordinate of the grid point */
//   double vx, vy, vz;             /* x, y and z component of the velocity field of the grid point */

// } GRIDPOINT;



// typedef struct EVALPOINTS {

//   bool   onray;             /* is true when the gridpoint is on any ray thus an evaluation point */

//   long   ray;                               /* number of the ray the evaluation point belongs to */
//   long    nr;                                    /* number of the evaluation point along the ray */

//   long   eqp;                                                         /* point on equivalent ray */

//   double  dZ;                                                /* distance increment along the ray */
//   double   Z;                                    /* distance between evaluation point and origin */
//   double vol;                  /* velocity along the ray between grid point and evaluation point */

// } EVALPOINT;



/* Declaration of external variables */


/* Grid and evaluation points */

extern long ngrid;                                                      /* number of grid points */
long ngrid;

extern long nsides;                                     /* determines the number of HEALPix rays */
long nsides;

extern long *cum_raytot;                /* cumulative number of evaluation points along each ray */
long *cum_raytot;

extern long *key;                  /* stores the numbers of the grid points on the rays in order */
long *key;

extern long *raytot;                    /* cumulative number of evaluation points along each ray */
long *raytot;


/* Level populations */

extern int nline_species;                                    /* number of line producing species */
int nline_species;

extern int *nlev;                                           /* number of levels for this species */
int *nlev;

extern int *nrad;                            /* number of radiative transitions for this species */
int *nrad;

extern int *cum_nlev;                                /* cumulative number of levels over species */
int *cum_nlev;

extern int *cum_nlev2;                           /* cumulative of squares of levels over species */
int *cum_nlev2;

extern int *cum_nrad;              /* cumulative of number of radiative transitions over species */
int *cum_nrad;

extern int tot_nlev;                                      /* total number of levels over species */
int tot_nlev;

extern int tot_nrad;                       /* total number of radiative transitions over species */
int tot_nrad;

extern int tot_nlev2;                          /* total number of squares of levels over species */
int tot_nlev2;


extern int *ncolpar;                            /* number of collision partners for this species */
int *ncolpar;

extern int *cum_ncolpar;                 /* cumulative number of collision partners over species */
int *cum_ncolpar;

extern int tot_ncolpar;                       /* total number of collision partners over species */
int tot_ncolpar;

extern int *ncoltemp;           /* number of collision temperatures for each species and partner */
int *ncoltemp;

extern int *ncoltran;          /* number of collisional transitions for each species and partner */
int *ncoltran;

extern int *cum_ncoltemp;       /* cumulative nr. of col. temperatures over species and partners */
int *cum_ncoltemp;

extern int *cum_ncoltran;        /* cumulative nr. of col. transitions over species and partners */
int *cum_ncoltran;

extern int *tot_ncoltemp;            /* total nr. of col. temperatures over species and partners */
int *tot_ncoltemp;

extern int *tot_ncoltran;             /* total nr. of col. transitions over species and partners */
int *tot_ncoltran;

extern int *cum_tot_ncoltemp;      /* cumulative of tot. of col. temp. over species and partners */
int *cum_tot_ncoltemp;

extern int *cum_tot_ncoltran;        /* cumulative tot. of col. trans. over species and partners */
int *cum_tot_ncoltran;

extern int *cum_ncoltrantemp;             /* cumulative of ntran*ntemp over species and partners */
int *cum_ncoltrantemp;

extern int *tot_ncoltrantemp;                  /* total of ntran*ntemp over species and partners */
int *tot_ncoltrantemp;

extern int *cum_tot_ncoltrantemp; /* cumulative of tot. of ntran*ntemp over species and partners */
int *cum_tot_ncoltrantemp;

extern int tot_cum_tot_ncoltran;            /* tot. cum. of tot. ntran over species and partners */
int tot_cum_tot_ncoltran;

extern int tot_cum_tot_ncoltemp;            /* tot. cum. of tot. ntemp over species and partners */
int tot_cum_tot_ncoltemp;

extern int tot_cum_tot_ncoltrantemp;  /* tot. cum. of tot. ntran*ntemp over species and partners */
int tot_cum_tot_ncoltrantemp;



/* ADDITIONS for the chemistry code */
/* -------------------------------- */

#define AU 1.66053878E-24                                                    /* atomic mass unit */


typedef struct SPECIES {

  std::string sym;                                                                 /* chemical symbol */

  double abn;                                                                       /* abundance */

  double mass;                                                                       /* mol mass */

} SPECIES;



typedef struct REACTIONS {

  std::string   R1;                                                                     /* reactant 1 */
  std::string   R2;                                                                     /* reactant 2 */
  std::string   R3;                                                                     /* reactant 3 */

  std::string   P1;                                                             /* reaction product 1 */
  std::string   P2;                                                             /* reaction product 2 */
  std::string   P3;                                                             /* reaction product 3 */
  std::string   P4;                                                             /* reaction product 4 */


double alpha;                               /* alpha coefficient to calculate rate coefficient k */
double beta;                                 /* beta coefficient to calculate rate coefficient k */
double gamma;                               /* gamma coefficient to calculate rate coefficient k */

double RT_min;                             /* RT_min coefficient to calculate rate coefficient k */
double RT_max;                             /* RT_max coefficient to calculate rate coefficient k */


double k;                                                           /* reaction rate coefficient */

int    dup;                                             /* Number of duplicates of this reaction */

} REACTIONS;


extern int nspec;                                                /* number of (chemical) species */
int nspec;

extern int nreac;                        /* number of chemical reactions in the chemical network */
int nreac;

/* -------------------------------- */



/*-----------------------------------------------------------------------------------------------*/
