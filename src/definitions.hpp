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



#ifndef __DEFINITIONS_HPP_INCLUDED__
#define __DEFINITIONS_HPP_INCLUDED__

#include <string>



/* Input and data files */

std::string grid_inputfile = GRID_INPUTFILE;    /* path to input file containing the grid points */

std::string spec_datafile = SPEC_DATAFILE;           /* path to data file containing the species */

std::string reac_datafile = REAC_DATAFILE;         /* path to data file containing the reactions */

std::string line_datafile[NLSPEC] = LINE_DATAFILES;              /* list of line data file paths */


// /* --- Addition by pre_setup --- */
//
//
// std::string line_datafile[NLSPEC] = { LINE_DATAFILE0, \
//                                       LINE_DATAFILE1, \
//                                       LINE_DATAFILE2, \
//                                       LINE_DATAFILE3  };
//
//
//  /* --- End of addition by pre_setup --- */



/* Declaration of external variables */


/* Grid and evaluation points */

long cum_raytot[NGRID*NRAYS];           /* cumulative number of evaluation points along each ray */

long key[NGRID*NGRID];             /* stores the numbers of the grid points on the rays in order */

long raytot[NGRID*NRAYS];               /* cumulative number of evaluation points along each ray */



/* Level populations */

int nlev[NLSPEC] = NLEV;                                    /* number of levels for this species */

int nrad[NLSPEC] = NRAD;                     /* number of radiative transitions for this species */


int cum_nlev[NLSPEC]  = CUM_NLEV;                    /* cumulative number of levels over species */

int cum_nlev2[NLSPEC] = CUM_NLEV2;               /* cumulative of squares of levels over species */

int cum_nrad[NLSPEC]  = CUM_NRAD;  /* cumulative of number of radiative transitions over species */


int ncolpar[NLSPEC]     = NCOLPAR;              /* number of collision partners for this species */

int cum_ncolpar[NLSPEC] = CUM_NCOLPAR;   /* cumulative number of collision partners over species */


int ncoltemp[TOT_NCOLPAR] = NCOLTEMP;    /* nr. of col. temperatures for each species & partners */

int ncoltran[TOT_NCOLPAR] = NCOLTRAN;     /* nr. of col. transitions for each species & partners */


int cum_ncoltemp[TOT_NCOLPAR] = CUM_NCOLTEMP;   /* cum. nr. of col. temps. over specs & partners */

int cum_ncoltran[TOT_NCOLPAR] = CUM_NCOLTRAN;   /* cum. nr. of col. trans. over specs & partners */

int cum_ncoltrantemp[TOT_NCOLPAR] = CUM_NCOLTRANTEMP;    /* cum. ntran*ntemp over specs & prtnrs */


int tot_ncoltemp[NLSPEC] = TOT_NCOLTEMP;     /* total nr. of col. temps. over species & partners */

int tot_ncoltran[NLSPEC] = TOT_NCOLTRAN;     /* total nr. of col. trans. over species & partners */

int tot_ncoltrantemp[NLSPEC] = TOT_NCOLTRANTEMP;  /* tot. of ntran*ntemp over species & partners */


int cum_tot_ncoltemp[NLSPEC] = CUM_TOT_NCOLTRAN; /* cum. tot. of col. temps. over specs & prtnrs */

int cum_tot_ncoltran[NLSPEC] = CUM_TOT_NCOLTRAN; /* cum. tot. of col. trans. over specs & prtnrs */

int cum_tot_ncoltrantemp[NLSPEC] = CUM_TOT_NCOLTRANTEMP; /* cumtot. ntran*ntemp o specs & prtnrs */



/* Chemistry */

SPECIES species[NSPEC];

REACTION reaction[NREAC];



int lspec_nr[NLSPEC];                                        /* nr of the line producing species */

int spec_par[TOT_NCOLPAR];         /* number of the species corresponding to a collision partner */

char ortho_para[TOT_NCOLPAR];                           /* stores whether it is ortho or para H2 */



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
