// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PARAMETERS_HPP_INCLUDED__
#define __PARAMETERS_HPP_INCLUDED__


#define RUN_NUMBER "0"

#define WRITE_INTERMEDIATE_OUTPUT false


// Input files

#define FIXED_NCELLS false
#define INPUT_FORMAT '.txt'


// VARIABLE GRID

#define GRID_INIT "input/files/1Dn30.dat_conv.txt"

#define X_MIN -1.0E+00
#define X_MAX +9.9E+99
#define Y_MIN -1.0E+00
#define Y_MAX +1.0E+00
#define Z_MIN -1.0E+00
#define Z_MAX +1.0E+00

#define THRESHOLD 1.0E+9;

#define SIZE_X 2
#define SIZE_Y 0
#define SIZE_Z 0



// Restart options

#define RESTART false
// #define RESTART_DIRECTORY "output/files/17-12-20_15:01_output/"

// #define INPUTFILE     "output/files/18-01-05_16:05_output/grid_reduced_0.1.txt"

#define INPUTFILE "input/files/1Dn30.dat_conv.txt"

#define SPEC_DATAFILE  "data/species_reduced.txt"
#define REAC_DATAFILE  "data/rates_reduced.txt"


#define NLSPEC 4

// #define LINE_DATAFILES {"data/12C+.txt", "data/16O.txt"}

#define LINE_DATAFILES { "data/12CO.txt", \
                         "data/12C.txt",  \
                         "data/12C+.txt", \
                         "data/16O.txt" }

// #define LINE_DATAFILES {"data/12C.txt", "data/12C+.txt", "data/16O.txt", "data/H2O.dat", "data/12CO.txt"}



// Ray tracing parameters

#define DIMENSIONS 1
#define NRAYS      2
#define NSIDES     6

#define THETA_CRIT 1.3

#define CELL_BASED true


// Radiative transfer

#define SOBOLEV                    true
#define ACCELERATION_POP_NG        true
#define ACCELERATION_APPROX_LAMBDA true


// Number of various iterations

#define MAX_NITERATIONS  299
#define PRELIM_CHEM_ITER 5
#define PRELIM_TB_ITER   20
#define CHEM_ITER        3


// Temperature range

#define TEMPERATURE_MIN T_CMB
#define TEMPERATURE_MAX 30000.0


// Chemistry

#define METALLICITY       1.0E+0
#define GAS_TO_DUST       1.0E+2
#define TIME_END_IN_YEARS 1.0E+7

#define ALWAYS_INITIALIZE_CHEMISTRY false


// External UV field

#define FIELD_FORM "UNI"

#define G_EXTERNAL_X 1.0E+1
#define G_EXTERNAL_Y 0.0E+0
#define G_EXTERNAL_Z 0.0E+0


// Turbulent velocity

#define V_TURB 1.0E5


// Cosmic ray variables

#define ZETA   3.846153846153846
#define OMEGA  0.42


#endif // __PARAMETERS_HPP_INCLUDED__
