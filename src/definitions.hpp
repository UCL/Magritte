// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __DEFINITIONS_HPP_INCLUDED__
#define __DEFINITIONS_HPP_INCLUDED__

#include <string>

#include "declarations.hpp"
#include "../setup/output_directory.hpp"


// Output directory

const std::string output_directory = OUTPUT_DIRECTORY;


// Input and data file paths (relative to Magritte folder)

// const std::string project_folder        = PROJECT_FOLDER;   // path to project folder

const std::string inputfile             = INPUTFILE;        // path to input file

// const std::string append_file           = APPEND_FILE;      // path to append file

const std::string spec_datafile         = SPEC_DATAFILE;    // path to data file with species

const std::string reac_datafile         = REAC_DATAFILE;    // path to data file with reactions

const std::string line_datafile[NLSPEC] = LINE_DATAFILES;   // list of line data file paths


// HEALPix vectors

const double healpixvector[3*NRAYS] = HEALPIXVECTOR;

const long antipod[NRAYS] = ANTIPOD;

// const long aligned[NRAYS][NRAYS/2] = ALIGNED;   // NOT USED YET

// const long n_aligned[NRAYS] = N_ALIGNED;        // NOT USED YET


// Level populations

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


// Chemistry

SPECIES species[NSPEC];

// REACTION reaction[NREAC];


int lspec_nr[NLSPEC];           // nr of line producing species

int spec_par[TOT_NCOLPAR];      // number of species corresponding to a collision partner

char ortho_para[TOT_NCOLPAR];   // stores whether it is ortho or para H2


// Species numbers

int nr_e;      // species nr corresponding to electrons

int nr_H2;     // species nr corresponding to H2

int nr_HD;     // species nr corresponding to HD

int nr_C;      // species nr corresponding to C

int nr_H;      // species nr corresponding to H

int nr_H2x;    // species nr corresponding to H2+

int nr_HCOx;   // species nr corresponding to HCO+

int nr_H3x;    // species nr corresponding to H3+

int nr_H3Ox;   // species nr corresponding to H3O+

int nr_Hex;    // species nr corresponding to He+

int nr_CO;     // species nr corresponding to CO


// Reaction numbers

int C_ionization_nr;

int H2_formation_nr;

int H2_photodissociation_nr;


#endif // __DEFINITIONS_HPP_INCLUDED__
