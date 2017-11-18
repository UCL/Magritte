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

const std::string grid_inputfile = GRID_INPUTFILE;    /* path to input file with the grid points */

const std::string spec_datafile = SPEC_DATAFILE;           /* path to data file with the species */

const std::string reac_datafile = REAC_DATAFILE;         /* path to data file with the reactions */

const std::string line_datafile[NLSPEC] = LINE_DATAFILES;        /* list of line data file paths */



/* Declaration of external variables */


/* HEALPix vectors */

double const unit_healpixvector[3*NRAYS] = UNIT_HEALPIXVECTOR;

long const antipod[NRAYS] = ANTIPOD;



/* Level populations */

int const nlev[NLSPEC] = NLEV;

int const nrad[NLSPEC] = NRAD;


int const cum_nlev[NLSPEC]  = CUM_NLEV;

int const cum_nlev2[NLSPEC] = CUM_NLEV2;

int const cum_nrad[NLSPEC]  = CUM_NRAD;


int const ncolpar[NLSPEC]     = NCOLPAR;

int const cum_ncolpar[NLSPEC] = CUM_NCOLPAR;


int const ncoltemp[TOT_NCOLPAR] = NCOLTEMP;

int const ncoltran[TOT_NCOLPAR] = NCOLTRAN;


int const cum_ncoltemp[TOT_NCOLPAR] = CUM_NCOLTEMP;

int const cum_ncoltran[TOT_NCOLPAR] = CUM_NCOLTRAN;

int const cum_ncoltrantemp[TOT_NCOLPAR] = CUM_NCOLTRANTEMP;


int const tot_ncoltemp[NLSPEC] = TOT_NCOLTEMP;

int const tot_ncoltran[NLSPEC] = TOT_NCOLTRAN;

int const tot_ncoltrantemp[NLSPEC] = TOT_NCOLTRANTEMP;


int const cum_tot_ncoltemp[NLSPEC] = CUM_TOT_NCOLTRAN;

int const cum_tot_ncoltran[NLSPEC] = CUM_TOT_NCOLTRAN;

int const cum_tot_ncoltrantemp[NLSPEC] = CUM_TOT_NCOLTRANTEMP;



/* Chemistry */

SPECIES species[NSPEC];

REACTION reaction[NREAC];



int lspec_nr[NLSPEC];                                        /* nr of the line producing species */

int spec_par[TOT_NCOLPAR];         /* number of the species corresponding to a collision partner */

char ortho_para[TOT_NCOLPAR];                           /* stores whether it is ortho or para H2 */



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





#endif /* __DEFINITIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
