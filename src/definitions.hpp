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



/* Input and data files */

string grid_inputfile = GRID_INPUTFILE;         /* path to input file containing the grid points */

string spec_datafile = SPEC_DATAFILE;                /* path to data file containing the species */

string reac_datafile = REAC_DATAFILE;              /* path to data file containing the reactions */


/* --- Addition by pre_setup --- */


string line_datafile[NLSPEC] = { LINE_DATAFILE0, \
                                 LINE_DATAFILE1, \
                                 LINE_DATAFILE2, \
                                 LINE_DATAFILE3  }; 
 

 /* --- End of addition by pre_setup --- */



/* Declaration of external variables */


/* Grid and evaluation points */

long cum_raytot[NGRID*NRAYS];           /* cumulative number of evaluation points along each ray */

long key[NGRID*NGRID];             /* stores the numbers of the grid points on the rays in order */

long raytot[NGRID*NRAYS];               /* cumulative number of evaluation points along each ray */



/* Level populations */

int nlev[NLSPEC];                                           /* number of levels for this species */

int nrad[NLSPEC];                            /* number of radiative transitions for this species */


int cum_nlev[NLSPEC];                                /* cumulative number of levels over species */

int cum_nlev2[NLSPEC];                           /* cumulative of squares of levels over species */

int cum_nrad[NLSPEC];              /* cumulative of number of radiative transitions over species */



int ncolpar[NLSPEC];                            /* number of collision partners for this species */

int cum_ncolpar[NLSPEC];                 /* cumulative number of collision partners over species */

int ncoltemp[TOT_NCOLPAR];           /* number of col. temperatures for each species and partner */

int ncoltran[TOT_NCOLPAR];            /* number of col. transitions for each species and partner */

int cum_ncoltemp[TOT_NCOLPAR];        /* cum. nr. of col. temperatures over species and partners */

int cum_ncoltran[TOT_NCOLPAR];         /* cum. nr. of col. transitions over species and partners */

int tot_ncoltemp[NLSPEC];            /* total nr. of col. temperatures over species and partners */

int tot_ncoltran[NLSPEC];             /* total nr. of col. transitions over species and partners */

int cum_tot_ncoltemp[NLSPEC];            /* cum. of tot. of col. temp. over species and partners */

int cum_tot_ncoltran[NLSPEC];        /* cumulative tot. of col. trans. over species and partners */

int cum_ncoltrantemp[TOT_NCOLPAR];        /* cumulative of ntran*ntemp over species and partners */

int tot_ncoltrantemp[NLSPEC];                  /* total of ntran*ntemp over species and partners */

int cum_tot_ncoltrantemp[NLSPEC];       /* cum. of tot. of ntran*ntemp over species and partners */




/* ----- ADDITIONS for the chemistry code -----                                                  */
/* --------------------------------------------------------------------------------------------- */

SPECIES species[NSPEC];

REACTION reaction[NREAC];


int spec_par[TOT_NCOLPAR];       /* number of the species corresponding to a collision partner */

char ortho_para[TOT_NCOLPAR];                         /* stores whether it is ortho or para H2 */


double metallicity;

double gas_to_dust;



/* Species numbers */

int e_nr;                                               /* species nr corresponding to electrons */

int H2_nr;                                                     /* species nr corresponding to H2 */

int HD_nr;                                                     /* species nr corresponding to HD */

int C_nr;                                                       /* species nr corresponding to C */

int H_nr;                                                       /* species nr corresponding to H */

int H2x_nr;                                                   /* species nr corresponding to H2+ */

int HCOx_nr;                                                 /* species nr corresponding to HCO+ */

int H3x_nr;                                                   /* species nr corresponding to H3+ */

int H3Ox_nr;                                                 /* species nr corresponding to H3O+ */

int Hex_nr;                                                   /* species nr corresponding to He+ */

int CO_nr;                                                     /* species nr corresponding to CO */


/* Reaction numbers */

int C_ionization_nr;

int H2_formation_nr;

int H2_photodissociation_nr;



double x_e;




#endif /* __DEFINITIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/


