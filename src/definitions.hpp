// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __DEFINITIONS_HPP_INCLUDED__
#define __DEFINITIONS_HPP_INCLUDED__

#include <string>

#include "declarations.hpp"
#include "../src/directories.hpp"


// Output directory (absolute path)

const std::string output_directory = OUTPUT_DIRECTORY;


// Absolute path to project folder

const std::string project_folder = PROJECT_FOLDER;


// Input file paths

const std::string inputfile_rel = INPUTFILE;                        // relative path
const std::string inputfile     = project_folder + inputfile_rel;   // absolute path


// Data file with species

const std::string spec_datafile_rel = SPEC_DATAFILE;                        // relative path
const std::string spec_datafile     = project_folder + spec_datafile_rel;   // absolute path


// Data file with reactions

const std::string reac_datafile_rel = REAC_DATAFILE;                        // relative path
const std::string reac_datafile     = project_folder + reac_datafile_rel;   // absolute path


const std::string line_datafile[NLSPEC] = LINE_DATAFILES;   // list of line data files


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

const int cum_ncoltemp[TOT_NCOLPAR]     = CUM_NCOLTEMP;
const int cum_ncoltran[TOT_NCOLPAR]     = CUM_NCOLTRAN;
const int cum_ncoltrantemp[TOT_NCOLPAR] = CUM_NCOLTRANTEMP;

const int tot_ncoltemp[NLSPEC]     = TOT_NCOLTEMP;
const int tot_ncoltran[NLSPEC]     = TOT_NCOLTRAN;
const int tot_ncoltrantemp[NLSPEC] = TOT_NCOLTRANTEMP;

const int cum_tot_ncoltemp[NLSPEC]     = CUM_TOT_NCOLTEMP;
const int cum_tot_ncoltran[NLSPEC]     = CUM_TOT_NCOLTRAN;
const int cum_tot_ncoltrantemp[NLSPEC] = CUM_TOT_NCOLTRANTEMP;


#endif // __DEFINITIONS_HPP_INCLUDED__
