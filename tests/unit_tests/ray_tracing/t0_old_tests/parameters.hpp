// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __PARAMETERS_HPP_INCLUDED__
#define __PARAMETERS_HPP_INCLUDED__



#define RUN_NUMBER "0"

#define WRITE_INTERMEDIATE_OUTPUT true

#define FIXED_NCELLS true
#define INPUT_FORMAT '.txt'
#define CELL_BASED  false

#define RESTART false


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

#define GRID_INIT "input/files/grid_2D_test_25.txt"

#define X_MIN  0.0E+00
#define X_MAX +5.0E+00
#define Y_MIN  0.0E+00
#define Y_MAX +5.0E+00
#define Z_MIN -1.0E+00
#define Z_MAX +1.0E+00

#define THRESHOLD 1.0E+99;

#define SIZE_X 9
#define SIZE_Y 9
#define SIZE_Z 0


// #define INPUTFILE "output/files/_output/grid.vtu"


// Input files

#define DIMENSIONS 2

#define NRAYS 8



// #define RESTART_DIRECTORY "output/files/17-12-20_15:01_output/"

// #define INPUTFILE     "output/files/18-01-05_16:05_output/grid_reduced_0.1.txt"

#define INPUTFILE "input/files/3DPDR_input/1Dn20.dat_conv.txt"


// #define PROJECT_FOLDER "input/files/Aori/"

// #define INPUT_FORMAT   '.vtu'

// #define INPUTFILE      "input/files/Aori/Aori_0001.vtu"

// #define APPEND_FILE    INPUTFILE

#define SPEC_DATAFILE  "data/species_reduced.txt"

#define REAC_DATAFILE  "data/rates_reduced.txt"


#define NLSPEC 4

#define LINE_DATAFILES {"data/12C.txt", "data/12C+.txt", "data/16O.txt", "data/12CO.txt"}


// #define LINE_DATAFILES {"data/12C.txt", "data/12C+.txt", "data/16O.txt", "data/H2O.dat", "data/12CO.txt"}



/* Ray tracing parameters */

#define NSIDES 6

#define THETA_CRIT 1.3

#define CELL_BASED true


/* Radiative transfer */

#define SOBOLEV true

#define ACCELERATION_POP_NG true

#define ACCELERATION_APPROX_LAMBDA true


/* Number of various iterations */

#define MAX_NITERATIONS 299

#define PRELIM_CHEM_ITER 0

#define PRELIM_TB_ITER 10

#define CHEM_ITER 8


/* Temperature range */

#define TEMPERATURE_MIN T_CMB

#define TEMPERATURE_MAX 30000.0


/* Chemistry */

#define METALLICITY 1.0

#define GAS_TO_DUST 100.0

#define TIME_END_IN_YEARS 1.0E+7

#define ALWAYS_INITIALIZE_CHEMISTRY true


/* External UV field */

#define FIELD_FORM "UNI"

#define G_EXTERNAL_X 5.270460E+0

#define G_EXTERNAL_Y 5.270460E+0

#define G_EXTERNAL_Z 5.666670E+0


/* Turbulent velocity */

#define V_TURB 1.0E5


/* Cosmic ray variables */

#define ZETA   3.846153846153846

#define OMEGA  0.42



#define NFREQ 4


// Helper constants

#define MAX_WIDTH 13      // for printing
#define BUFFER_SIZE 3500  // max number of characters in a line


// Parameters for level population iteration

#define POP_PREC        1.0E-2    // precision used in convergence criterion
#define POP_LOWER_LIMIT 1.0E-26   // lowest non-zero population
#define POP_UPPER_LIMIT 1.0E+15   // highest population
#define TAU_MAX         3.0E+2    // cut-off for optical depth along a ray


// Parameters for thermal balance iteration

#define THERMAL_PREC 1.0E-3   // precision used in convergence criterion


#endif /* __PARAMETERS_HPP_INCLUDED__ */