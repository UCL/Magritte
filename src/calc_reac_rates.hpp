// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __calc_reac_rates_HPP_INCLUDED__
#define __calc_reac_rates_HPP_INCLUDED__


// rate_H2_formation: returns rate coefficient for H2 formation reaction
// ---------------------------------------------------------------------

double rate_H2_formation (REACTION *reaction, int reac, double temperature_gas, double temperature_dust);


// rate_PAH: returns rate coefficient for reactions with PAHs
// ----------------------------------------------------------

double rate_PAH (REACTION *reaction, int reac, double temperature_gas);


// rate_CRP: returns rate coefficient for reaction induced by cosmic rays
// ----------------------------------------------------------------------

double rate_CRP (REACTION *reaction, int reac, double temperature_gas);


// rate_CRPHOT: returns rate coefficient for reaction induced by cosmic rays
// -------------------------------------------------------------------------

double rate_CRPHOT (REACTION *reaction, int reac, double temperature_gas);


// rate_FREEZE: returns rate coefficient for freeze-out reaction of neutral species
// --------------------------------------------------------------------------------

double rate_FREEZE (REACTION *reaction, int reac, double temperature_gas);


// rate_ELFRZE: returns rate coefficient for freeze-out reaction of singly charged positive ions
// ---------------------------------------------------------------------------------------------

double rate_ELFRZE (REACTION *reaction, int reac, double temperature_gas);


// rate_CRH: returns rate coefficient for desorption due to cosmic ray heating
// ---------------------------------------------------------------------------

double rate_CRH (REACTION *reaction, int reac, double temperature_gas);


// rate_THERM: returns rate coefficient for thermal desorption
// -----------------------------------------------------------

double rate_THERM (REACTION *reaction, int reac, double temperature_gas, double temperature_dust);


// rate_GM: returns rate coefficient for grain mantle reactions
// ------------------------------------------------------------

double rate_GM (REACTION *reaction, int reac);


// rate_canonical: returns canonical rate coefficient for the reaction
// -------------------------------------------------------------------

double rate_canonical (REACTION *reaction, int reac, double temperature_gas);


#endif // __calc_reac_rates_HPP_INCLUDED__
