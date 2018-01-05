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

std::string output_directory = OUTPUT_DIRECTORY;


// Input and data files

const std::string inputfile             = INPUTFILE;        // path to input file

const std::string append_file           = APPEND_FILE;      // path to input file

const std::string spec_datafile         = SPEC_DATAFILE;    // path to data file with species

const std::string reac_datafile         = REAC_DATAFILE;    // path to data file with reactions

const std::string line_datafile[NLSPEC] = LINE_DATAFILES;   // list of line data file paths


// HEALPix vectors

const double healpixvector[3*NRAYS] = HEALPIXVECTOR;

const long antipod[NRAYS] = ANTIPOD;


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


// Roots and weights for Gauss-Hermite quadrature

const double H_4_weights[4] = WEIGHTS_4;   // weights for 4th order Gauss-Hermite quadrature
const double H_4_roots[4]   = ROOTS_4;     // roots of 4th (physicists') Hermite polynomial

const double H_5_weights[5] = WEIGHTS_5;   // weights for 5th order Gauss-Hermite quadrature
const double H_5_roots[5]   = ROOTS_5;     // roots of 5th (physicists') Hermite polynomial

const double H_7_weights[7] = WEIGHTS_7;   // weights for 7th order Gauss-Hermite quadrature
const double H_7_roots[7]   = ROOTS_7;     // roots of 7th (physicists') Hermite polynomial


// Chemistry

SPECIES species[NSPEC];

REACTION reaction[NREAC];


int lspec_nr[NLSPEC];           // nr of line producing species

int spec_par[TOT_NCOLPAR];      // number of species corresponding to a collision partner

char ortho_para[TOT_NCOLPAR];   // stores whether it is ortho or para H2


// Species numbers

int e_nr;      // species nr corresponding to electrons

int H2_nr;     // species nr corresponding to H2

int HD_nr;     // species nr corresponding to HD

int C_nr;      // species nr corresponding to C

int H_nr;      // species nr corresponding to H

int H2x_nr;    // species nr corresponding to H2+

int HCOx_nr;   // species nr corresponding to HCO+

int H3x_nr;    // species nr corresponding to H3+

int H3Ox_nr;   // species nr corresponding to H3O+

int Hex_nr;    // species nr corresponding to He+

int CO_nr;     // species nr corresponding to CO


// Reaction numbers

int C_ionization_nr;

int H2_formation_nr;

int H2_photodissociation_nr;


#endif // __DEFINITIONS_HPP_INCLUDED__
