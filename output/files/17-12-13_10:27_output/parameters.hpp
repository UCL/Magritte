/* Parameters file */

#ifndef __PARAMETERS_HPP_INCLUDED__
#define __PARAMETERS_HPP_INCLUDED__



#define RUN_NUMBER "1"

#define WRITE_INTERMEDIATE_OUTPUT false

#define d d

/* Input files */

#define INPUT_FORMAT TXT

#define GRID_INPUTFILE "input/1Dn30.dat_conv.txt"

// #define INPUT_FORMAT VTU

// #define GRID_INPUTFILE "input/Aori_0001.vtu"

#define SPEC_DATAFILE  "data/species_reduced.txt"

#define REAC_DATAFILE  "data/rates_reduced.txt"

#define NLSPEC 4

#define LINE_DATAFILES {"data/12C.txt", "data/12C+.txt", "data/16O.txt", "data/12CO.txt"}


// #define LINE_DATAFILES {"data/12C.txt", "data/12C+.txt", "data/16O.txt", "data/H2O.dat", "data/12CO.txt"}



/* Ray tracing parameters */

#define NSIDES 1

#define THETA_CRIT 1.3

#define RAY_SEPARATION2 0.0

#define ON_THE_FLY true


/* Radiative transfer */

#define SOBOLEV true

#define ACCELERATION_POP_NG true

#define ACCELERATION_APPROX_LAMBDA true


/* Number of various iterations */

#define MAX_NITERATIONS 0

#define PRELIM_CHEM_ITER 5

#define PRELIM_TB_ITER 0

#define CHEM_ITER 0


/* Temperature range */

#define TEMPERATURE_MIN T_CMB

#define TEMPERATURE_MAX 30000.0


/* Chemistry */

#define METALLICITY 1.0

#define GAS_TO_DUST 100.0;

#define TIME_END_IN_YEARS 1.0E+7

#define ALWAYS_INITIALIZE_CHEMISTRY false


/* External UV field */

#define FIELD_FORM "UNI"

#define G_EXTERNAL_X 3.270460E+1

#define G_EXTERNAL_Y 3.270460E+1

#define G_EXTERNAL_Z 3.666670E+1


/* Turbulent velocity */

#define V_TURB 1.0E5


/* Cosmic ray variables */

#define ZETA   3.846153846153846

#define OMEGA  0.42



#endif /* __PARAMETERS_HPP_INCLUDED__ */
