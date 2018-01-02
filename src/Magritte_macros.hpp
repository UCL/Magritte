// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#define GINDEX(r,c)   ( (c) + (r)*NCELLS )   // when second index is grid point

#define RINDEX(r,c)   ( (c) + (r)*NRAYS )    // when second index is ray

#define VINDEX(r,c)   ( (c) + (r)*3 )        // when second index is a 3-vector index


// Special key to find the number of the grid point associated to a certain evaluation point

#define LOCAL_GP_NR_OF_EVALP(ray, evalp)   ( key[ (evalp) + cum_raytot[(ray)] ] )
// gives grid point number corresponding to "evalp"'th evaluation point on ray "ray"


/* Level population related index definitions */

#define LSPECLEV(lspec,i)   ( (i) + cum_nlev[(lspec)] )                                           \
                               /* when first index is line producing species and second is level */

#define LSPECLEVLEV(lspec,i,j)   ( (j) + (i)*nlev[(lspec)] + cum_nlev2[(lspec)] )                 \
                   /* when first index is line producing species and second and third are levels */

#define LSPECGRIDLEVLEV(lspec,gridp,i,j)   ( (j) + (i)*nlev[(lspec)]                              \
                                             + (gridp)*nlev[(lspec)]*nlev[(lspec)]                \
                                             + NCELLS*cum_nlev2[(lspec)] )                         \
/* when first index is line producing species, second is grid point, third and fourth are levels */

#define LSPECGRIDLEV(lspec,gridp,i)   ( (i) + (gridp)*nlev[(lspec)] + NCELLS*cum_nlev[lspec] )     \
          /* when first index is line producing species, second is grid point and third is level */

#define LSPECRAD(lspec,kr)   ( (kr) + cum_nrad[(lspec)] )                                         \
                /* when first index is line producing species and second is radiative transition */

#define LSPECGRIDRAD(lspec,gridp,kr)   ( (kr) + (gridp)*nrad[(lspec)] + NCELLS*cum_nrad[(lspec)] )
/* when first index is line producing species, second is grid point and third is rad. transition */


#define LINDEX(i,j) ((j)+(i)*nlev[lspec])                        /* when second index are levels */

#define L2INDEX(r,c) ((c)+(r)*nlev[lspec]*nlev[lspec])       /* when second index is LINDEX(i,j) */

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
