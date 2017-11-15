/* Parameters file */


/* Input files */

#define GRID_INPUTFILE "input/1Dn30.dat_conv.txt"

#define SPEC_DATAFILE  "data/species_reduced.txt"

#define REAC_DATAFILE  "data/rates_reduced.txt"

#define NLSPEC 4

#define LINE_DATAFILES {"data/12C.txt", "data/12C+.txt", "data/16O.txt", "data/12CO.txt"}


/* Ray tracing parameters */

#define NSIDES 1

#define THETA_CRIT 1.300000

#define RAY_SEPARATION2 0.000000E+00


/* Radiative transfer */

#define SOBOLEV true

#define ACCELERATION_POP_NG true


/* Number of various iterations */

#define MAX_NITERATIONS 300

#define PRELIM_CHEM_ITER 5

#define PRELIM_TB_ITER 10

#define CHEM_ITER 3


/* Temperature range */

#define TEMPERATURE_MIN T_CMB

#define TEMPERATURE_MAX 30000.0


/* Chemistry */

#define METALLICITY 1.0

#define GAS_TO_DUST 100.0;

#define TIME_END_IN_YEARS 1.000000E+07


/* External UV field */

#define FIELD_FORM "UNI"

#define G_EXTERNAL_X 5.270460E+00

#define G_EXTERNAL_Y 5.270460E+00

#define G_EXTERNAL_Z 6.666670E+00


/* Turbulent velocity */

#define V_TURB 1.0E5