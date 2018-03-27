// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


// General index definitions

#define RINDEX(r,c)   ( (c) + (r)*NRAYS )   // when second index is ray
#define SINDEX(r,c)   ( (c) + (r)*NSPEC )   // when second index is species
#define READEX(r,c)   ( (c) + (r)*NREAC )   // when second index is reaction

#define LINDEX(r,c)   ( (c) + (r)*TOT_NLEV )   // when second index is level
#define KINDEX(r,c)   ( (c) + (r)*TOT_NRAD )   // when second index is transition


#define LLINDEX(ls,i,j)   ( (j) + (i)*nlev[(ls)] )   // when second index are levels


// Line related index definitions

#define LSPECLEV(lspec,i)   ( (i) + cum_nlev[(lspec)] )                                           \
                               /* when first index is line producing species and second is level */

#define LSPECLEVLEV(lspec,i,j)   ( (j) + (i)*nlev[(lspec)] + cum_nlev2[(lspec)] )                 \
                   /* when first index is line producing species and second and third are levels */

#define LSPECGRIDLEVLEV(lspec,o,i,j)   ( (j) + (i)*nlev[(lspec)]                              \
                                             + (o)*nlev[(lspec)]*nlev[(lspec)]                \
                                             + NCELLS*cum_nlev2[(lspec)] )                        \
/* when first index is line producing species, second is grid point, third and fourth are levels */

#define LSPECGRIDLEV(lspec,o,i)   ( (i) + (o)*nlev[(lspec)] + NCELLS*cum_nlev[lspec] )    \
          /* when first index is line producing species, second is grid point and third is level */

#define LSPECRAD(lspec,kr)   ( (kr) + cum_nrad[(lspec)] )                                         \
                /* when first index is line producing species and second is radiative transition */

#define LSPECGRIDRAD(lspec,o,kr)   ( (kr) + (o)*nrad[(lspec)] + NCELLS*cum_nrad[(lspec)] )
/* when first index is line producing species, second is grid point and third is rad. transition */


// Collision rate related indices

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
