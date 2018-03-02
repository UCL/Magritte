// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PARAMETERS_HPP_INCLUDED__
#define __PARAMETERS_HPP_INCLUDED__


#define RUN_NUMBER "0"
#define WRITE_INTERMEDIATE_OUTPUT false


// Input files

#define FIXED_NCELLS true
#define INPUT_FORMAT '.vtu'
#define CELL_BASED  false

#define NAME_DENSITY              "H2_density"
#define NAME_VELOCITY             "velocity"
#define NAME_VX                   "v1"
#define NAME_VY                   "v2"
#define NAME_VZ                   "v3"
#define NAME_TEMPERATURE_GAS      "Gas_temperature"
#define NAME_TEMPERATURE_DUST     "Gas_temperature"
#define NAME_TEMPERATURE_GAS_PREV "Gas_temperature"
#define NAME_CHEM_ABUNDANCES      "Mol_density"


// VARIABLE GRID

#define GRID_INIT "input/files/grid_log3.vtk"

#define X_MIN -7.5E+14
#define X_MAX +7.5E+14
#define Y_MIN -7.5E+14
#define Y_MAX +7.5E+14
#define Z_MIN -7.5E+14
#define Z_MAX +7.5E+14

#define THRESHOLD 1.0E+99;

#define SIZE_X 25
#define SIZE_Y 25
#define SIZE_Z 25


// Restart options

#define RESTART true
#define RESTART_DIRECTORY "input/"



// #define INPUTFILE     "output/files/18-01-05_16:05_output/grid_reduced_0.1.txt"

#define INPUTFILE      "input/files/grid_log3.vtu"

#define SPEC_DATAFILE  "data/test_species.txt"
#define REAC_DATAFILE  "data/test_rates.txt"


#define NLSPEC 1

#define LINE_DATAFILES { "data/12CO.txt" }

// #define LINE_DATAFILES {"data/12C.txt", "data/12C+.txt", "data/16O.txt", "data/H2O.dat", "data/12CO.txt"}


// Ray tracing parameters

#define DIMENSIONS 3
// #define NRAYS      2

// NRAYS = 12 * NSIDES * NSIDES
#define NSIDES     1


// Radiative transfer

#define SOBOLEV                    false
#define ACCELERATION_POP_NG        true
#define ACCELERATION_APPROX_LAMBDA true


// Number of various iterations

#define MAX_NITERATIONS  150
#define PRELIM_CHEM_ITER 0
#define PRELIM_TB_ITER   0
#define CHEM_ITER        0


// Temperature range

#define TEMPERATURE_MIN T_CMB
#define TEMPERATURE_MAX 30000.0


// Chemistry

#define METALLICITY       1.0E+0
#define GAS_TO_DUST       1.0E+2
#define TIME_END_IN_YEARS 1.0E+7

#define ALWAYS_INITIALIZE_CHEMISTRY false


// External UV field

#define FIELD_FORM "ISO"

#define G_EXTERNAL_X 0.0E+0
#define G_EXTERNAL_Y 0.0E+0
#define G_EXTERNAL_Z 0.0E+0


// Turbulent velocity

#define V_TURB 1.0E5


// Cosmic ray variables

#define ZETA   3.846153846153846
#define OMEGA  0.42

#define NFREQ 4


// Helper constants

#define MAX_WIDTH     13  // for printing
#define BUFFER_SIZE 3500  // max number of characters in a line


// Parameters for level population iteration

#define POP_PREC        1.0E-2    // precision used in convergence criterion
#define POP_LOWER_LIMIT 1.0E-26   // lowest non-zero population
#define POP_UPPER_LIMIT 1.0E+15   // highest population
#define TAU_MAX         3.0E+2    // cut-off for optical depth along a ray


// Parameters for thermal balance iteration

#define THERMAL_PREC 1.0E-3   // precision used in convergence criterion


#endif // __PARAMETERS_HPP_INCLUDED__
