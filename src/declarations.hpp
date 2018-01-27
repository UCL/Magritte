// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __DECLARATIONS_HPP_INCLUDED__
#define __DECLARATIONS_HPP_INCLUDED__


#include <string>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "Magritte_constants.hpp"
#include "Magritte_macros.hpp"
#include "Magritte_types.hpp"




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


// Output directory

extern const std::string output_directory;

// extern const std::string project_folder;


// HEALPix vectors

extern const double healpixvector[3*NRAYS];

extern const long antipod[NRAYS];

// extern const long aligned[NRAYS][NRAYS/2];   // NOT USED YET

// extern const long n_aligned[NRAYS];          // NOT USED YET



/* Level populations */

extern const int nlev[NLSPEC];                              /* number of levels for this species */

extern const int nrad[NLSPEC];               /* number of radiative transitions for this species */

extern const int cum_nlev[NLSPEC];                   /* cumulative number of levels over species */

extern const int cum_nlev2[NLSPEC];              /* cumulative of squares of levels over species */

extern const int cum_nrad[NLSPEC]; /* cumulative of number of radiative transitions over species */



extern const int ncolpar[NLSPEC];               /* number of collision partners for this species */

extern const int cum_ncolpar[NLSPEC];    /* cumulative number of collision partners over species */

extern const int ncoltemp[TOT_NCOLPAR];    /* number of col. temperatures for each specs & prtnr */

extern const int ncoltran[TOT_NCOLPAR];     /* number of col. transitions for each specs & prtnr */

extern const int cum_ncoltemp[TOT_NCOLPAR]; /* cum. nr. of col. temperatures over specs & prtnrs */

extern const int cum_ncoltran[TOT_NCOLPAR];  /* cum. nr. of col. transitions over specs & prtnrs */

extern const int tot_ncoltemp[NLSPEC];     /* total nr. of col. temperatures over specs & prtnrs */

extern const int tot_ncoltran[NLSPEC];      /* total nr. of col. transitions over specs & prtnrs */

extern const int cum_tot_ncoltemp[NLSPEC];     /* cum. of tot. of col. temp. over specs & prtnrs */

extern const int cum_tot_ncoltran[NLSPEC]; /* cumulative tot. of col. trans. over specs & prtnrs */

extern const int cum_ncoltrantemp[TOT_NCOLPAR]; /* cumulative of ntran*ntemp over specs & prtnrs */

extern const int tot_ncoltrantemp[NLSPEC];           /* total of ntran*ntemp over specs & prtnrs */

extern const int cum_tot_ncoltrantemp[NLSPEC];   /* cum. of tot. of ntran*ntemp o specs & prtnrs */


// Magritte constants

extern const double PI;      // pi
extern const double CC;      // speed of light in cgs units
extern const double HH;      // Planck's constant in cgs units
extern const double KB;      // Boltzmann's constant in cgs units
extern const double EV;      // one electron Volt in erg
extern const double MP;      // proton mass in cgs units
extern const double PC;      // one parsec in cm
extern const double AU;      // atomic mass unit
extern const double T_CMB;   // CMB  temperature in K

extern const double SECONDS_IN_YEAR;   // number of seconds in one year


// Roots and weights for Gauss-Hermite quadrature

extern const double H_4_weights[4];   // weights for 4th order Gauss-Hermite quadrature
extern const double H_4_roots[4];     // roots of 4th (physicists') Hermite polynomial

extern const double H_5_weights[5];   // weights for 5th order Gauss-Hermite quadrature
extern const double H_5_roots[5];     // roots of 5th (physicists') Hermite polynomial

extern const double H_7_weights[7];   // weights for 7th order Gauss-Hermite quadrature
extern const double H_7_roots[7];     // roots of 7th (physicists') Hermite polynomial



// Chemistry


// Species numbers

extern int nr_e;      // species nr corresponding to electrons
extern int nr_H2;     // species nr corresponding to H2
extern int nr_HD;     // species nr corresponding to HD
extern int nr_C;      // species nr corresponding to C
extern int nr_H;      // species nr corresponding to H
extern int nr_H2x;    // species nr corresponding to H2+
extern int nr_HCOx;   // species nr corresponding to HCO+
extern int nr_H3x;    // species nr corresponding to H3+
extern int nr_H3Ox;   // species nr corresponding to H3O+
extern int nr_Hex;    // species nr corresponding to He+
extern int nr_CO;     // species nr corresponding to CO


// Reaction numbers

extern int nr_C_ionization;
extern int nr_H2_formation;
extern int nr_H2_photodissociation;


#endif // __DECLARATIONS_HPP_INCLUDED__
