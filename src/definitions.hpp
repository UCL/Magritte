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

#include "../setup/output_directory.hpp"



/* Output directory */

const std::string output_directory = OUTPUT_DIRECTORY;



/* Input and data files */

const std::string grid_inputfile = GRID_INPUTFILE;    /* path to input file with the grid points */

const std::string spec_datafile = SPEC_DATAFILE;           /* path to data file with the species */

const std::string reac_datafile = REAC_DATAFILE;         /* path to data file with the reactions */

const std::string line_datafile[NLSPEC] = LINE_DATAFILES;        /* list of line data file paths */



/* Declaration of external variables */


/* HEALPix vectors */

const double healpixvector[3*NRAYS] = HEALPIXVECTOR;

const long antipod[NRAYS] = ANTIPOD;



/* Level populations */

const int nlev[NLSPEC] = NLEV;

const int nrad[NLSPEC] = NRAD;


const int cum_nlev[NLSPEC]  = CUM_NLEV;

const int cum_nlev2[NLSPEC] = CUM_NLEV2;

const int cum_nrad[NLSPEC]  = CUM_NRAD;


const int ncolpar[NLSPEC]     = NCOLPAR;

const int cum_ncolpar[NLSPEC] = CUM_NCOLPAR;


const int ncoltemp[TOT_NCOLPAR] = NCOLTEMP;

const int ncoltran[TOT_NCOLPAR] = NCOLTRAN;


const int cum_ncoltemp[TOT_NCOLPAR] = CUM_NCOLTEMP;

const int cum_ncoltran[TOT_NCOLPAR] = CUM_NCOLTRAN;

const int cum_ncoltrantemp[TOT_NCOLPAR] = CUM_NCOLTRANTEMP;


const int tot_ncoltemp[NLSPEC] = TOT_NCOLTEMP;

const int tot_ncoltran[NLSPEC] = TOT_NCOLTRAN;

const int tot_ncoltrantemp[NLSPEC] = TOT_NCOLTRANTEMP;


const int cum_tot_ncoltemp[NLSPEC] = CUM_TOT_NCOLTRAN;

const int cum_tot_ncoltran[NLSPEC] = CUM_TOT_NCOLTRAN;

const int cum_tot_ncoltrantemp[NLSPEC] = CUM_TOT_NCOLTRANTEMP;



/* Roots of the 5th (physicists') Hermite polynomial */

const double H_4_weights[NFREQ] = WEIGHTS_4;

const double H_4_roots[NFREQ] = ROOTS_4;



/* Roots of the 5th (physicists') Hermite polynomial */

// const double H_5_weights[NFREQ] = WEIGHTS_5;
//
// const double H_5_roots[NFREQ] = ROOTS_5;



/* Roots of the 7th (physicists') Hermite polynomial */

// const double H_7_weights[NFREQ] = WEIGHTS_7;
//
// const double H_7_roots[NFREQ] = ROOTS_7;



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
