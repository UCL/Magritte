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
#define BUFFER_SIZE 500   // max number of characters in a line


// Parameters for level population iteration

#define POP_PREC        1.0E-2    // precision used in convergence criterion
#define POP_LOWER_LIMIT 1.0E-26   // lowest non-zero population
#define POP_UPPER_LIMIT 1.0E+15   // highest population
#define TAU_MAX         1.0E+3    // cut-off for optical depth along a ray


// Parameters for thermal balance iteration

#define THERMAL_PREC 1.0E-3   // precision used in convergence criterion


// Output directory

extern std::string output_directory;


// HEALPix vectors

extern const double healpixvector[3*NRAYS];

extern const long antipod[NRAYS];



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



// Roots and weights for Gauss-Hermite quadrature

extern const double H_4_weights[4];   // weights for 4th order Gauss-Hermite quadrature
extern const double H_4_roots[4];     // roots of 4th (physicists') Hermite polynomial

extern const double H_5_weights[5];   // weights for 5th order Gauss-Hermite quadrature
extern const double H_5_roots[5];     // roots of 5th (physicists') Hermite polynomial

extern const double H_7_weights[7];   // weights for 7th order Gauss-Hermite quadrature
extern const double H_7_roots[7];     // roots of 7th (physicists') Hermite polynomial



// Chemistry

extern SPECIES species[NSPEC];

extern REACTION reaction[NREAC];


extern int lspec_nr[NLSPEC];           // names of line producing species

extern int spec_par[TOT_NCOLPAR];      // number of species corresponding to a collision partner

extern char ortho_para[TOT_NCOLPAR];   // stores whether it is ortho or para H2


// Species numbers

extern int e_nr;      // species nr corresponding to electrons
extern int H2_nr;     // species nr corresponding to H2
extern int HD_nr;     // species nr corresponding to HD
extern int C_nr;      // species nr corresponding to C
extern int H_nr;      // species nr corresponding to H
extern int H2x_nr;    // species nr corresponding to H2+
extern int HCOx_nr;   // species nr corresponding to HCO+
extern int H3x_nr;    // species nr corresponding to H3+
extern int H3Ox_nr;   // species nr corresponding to H3O+
extern int Hex_nr;    // species nr corresponding to He+
extern int CO_nr;     // species nr corresponding to CO


// Reaction numbers

extern int C_ionization_nr;
extern int H2_formation_nr;
extern int H2_photodissociation_nr;


#endif // __DECLARATIONS_HPP_INCLUDED__
