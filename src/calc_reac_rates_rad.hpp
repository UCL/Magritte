// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __calc_reac_rates_rad_HPP_INCLUDED__
#define __calc_reac_rates_rad_HPP_INCLUDED__


/* Note in the arguments that the temperatures are local (doubles), but rad_surface, AV and column
   densities are still represented by the pointers to the full arrays */


// rate_PHOTD: returns rate coefficient for photodesorption
// --------------------------------------------------------

double rate_PHOTD (CELL *cell, REACTION *reaction, int reac, long o);


// rate_H2_photodissociation: returns rate coefficient for H2 dissociation
// -----------------------------------------------------------------------

double rate_H2_photodissociation (CELL *cell, REACTION *reaction, int reac, double *column_H2, long o);


// rate_CO_photodissociation: returns rate coefficient for CO dissociation
// -----------------------------------------------------------------------

double rate_CO_photodissociation (CELL *cell, REACTION *reaction, int reac,
                                  double *column_CO, double *column_H2, long o);


// rate_C_photoionization: returns rate coefficient for C photoionization
// ----------------------------------------------------------------------

double rate_C_photoionization (CELL *cell, REACTION *reaction, int reac,
                               double *column_C, double *column_H2, long o);


// rate_SI_photoionization: returns rate coefficient for SI photoionization
// ------------------------------------------------------------------------

double rate_SI_photoionization (CELL *cell, REACTION *reaction, int reac, long o);


// rate_canonical_photoreaction: returns rate coefficient for a canonical photoreaction
// ------------------------------------------------------------------------------------

double rate_canonical_photoreaction (CELL *cell, REACTION *reaction, int reac, long o);


#endif // __calc_reac_rates_rad_HPP_INCLUDED__
