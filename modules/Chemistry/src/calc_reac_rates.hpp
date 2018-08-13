// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __calc_reac_rates_HPP_INCLUDED__
#define __calc_reac_rates_HPP_INCLUDED__


// rate_H2_formation: returns rate coefficient for H2 formation reaction
// ---------------------------------------------------------------------

double rate_H2_formation (CELLS *cells, REACTIONS reactions, int e, long o);


// rate_PAH: returns rate coefficient for reactions with PAHs
// ----------------------------------------------------------

double rate_PAH (CELLS *cells, REACTIONS reactions, int e, long o);


// rate_CRP: returns rate coefficient for reaction induced by cosmic rays
// ----------------------------------------------------------------------

double rate_CRP (CELLS *cells, REACTIONS reactions, int e, long o);


// rate_CRPHOT: returns rate coefficient for reaction induced by cosmic rays
// -------------------------------------------------------------------------

double rate_CRPHOT (CELLS *cells, REACTIONS reactions, int e, long o);


// rate_FREEZE: returns rate coefficient for freeze-out reaction of neutral species
// --------------------------------------------------------------------------------

double rate_FREEZE (CELLS *cells, REACTIONS reactions, int e, long o);


// rate_ELFRZE: returns rate coefficient for freeze-out reaction of singly charged positive ions
// ---------------------------------------------------------------------------------------------

double rate_ELFRZE (CELLS *cells, REACTIONS reactions, int e, long o);


// rate_CRH: returns rate coefficient for desorption due to cosmic ray heating
// ---------------------------------------------------------------------------

double rate_CRH (REACTIONS reactions, int e);


// rate_THERM: returns rate coefficient for thermal desorption
// -----------------------------------------------------------

double rate_THERM (CELLS *cells, REACTIONS reactions, int e, long o);


// rate_GM: returns rate coefficient for grain mantle reactions
// ------------------------------------------------------------

double rate_GM (REACTIONS reactions, int e);


// rate_canonical: returns canonical rate coefficient for the reaction
// -------------------------------------------------------------------

double rate_canonical (CELLS *cells, REACTIONS reactions, int e, long o);


#endif // __calc_reac_rates_HPP_INCLUDED__
