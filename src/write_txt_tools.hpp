// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRITE_TXT_TOOLS_HPP_INCLUDED__
#define __WRITE_TXT_TOOLS_HPP_INCLUDED__


#include <string>

#include "declarations.hpp"


// write_grid: write grid again
// ----------------------------

int write_grid (std::string tag, CELL *cell);


// write_healpixvectors: write unit HEALPix vectors
// ------------------------------------------------

int write_healpixvectors (std::string tag);


// write_eval: Write evaluation points (Z along ray and number of ray)
// -------------------------------------------------------------------

int write_eval (std::string tag, EVALPOINT *evalpoint);


// write_key: write key to find which grid point corresponds to which evaluation point
// -----------------------------------------------------------------------------------

int write_key (std::string tag, long *key);


// write_raytot: write total of evaluation points along each ray
// -------------------------------------------------------------

int write_raytot (std::string tag, long *raytot);


// write_cum_raytot: write cumulative total of evaluation points along each ray
// ----------------------------------------------------------------------------

int write_cum_raytot (std::string tag, long *cum_raytot);


// write_abundances: write abundances at each point
// ------------------------------------------------

int write_abundances (std::string tag);


// write_level_populations: write level populations at each point for each transition
// ----------------------------------------------------------------------------------

int write_level_populations (std::string tag, double *pop);


// write_line_intensities: write line intensities for each species, point and transition
// -------------------------------------------------------------------------------------

int write_line_intensities (std::string tag, double *mean_intensity);


// write_temperature_gas: write gas temperatures at each point
// -----------------------------------------------------------

int write_temperature_gas (std::string tag, double *temperature_gas);


// write_temperature_dust: write dust temperatures at each point
// -------------------------------------------------------------

int write_temperature_dust (std::string tag, double *temperature_dust);


// write_prev_temperature_gas: write previous gas temperatures at each point
//--------------------------------------------------------------------------

int write_prev_temperature_gas (std::string tag, double *prev_temperature_gas);


// write_UV_field: write UV field at each point
// --------------------------------------------

int write_UV_field (std::string tag, double *UV_field);


// write_UV_field: write UV field at each point
// --------------------------------------------

int write_AV (std::string tag, double *AV);


// write_rad_surface: write rad surface at each point
// --------------------------------------------------

int write_rad_surface (std::string tag, double *rad_surface);


// write_reaction_rates: write rad surface at each point
// -----------------------------------------------------

int write_reaction_rates (std::string tag, REACTION *reaction);


// write_certain_reactions: write rates of certain reactions
// ---------------------------------------------------------

int write_certain_rates (std::string tag, std::string name, int nr_certain_reac,
                         int *certain_reactions, REACTION *reaction);


// write_double_1: write a 1D list of doubles
// ------------------------------------------

int write_double_1 (std::string name, std::string tag, long length, double *variable);


// write_double_2: write a 2D array of doubles
// -------------------------------------------

int write_double_2 (std::string name, std::string tag, long nrows, long ncols, double *variable);


// write_radfield_tools: write output of functoins defined in radfield_tools
// -------------------------------------------------------------------------

int write_radfield_tools (std::string tag, double *AV ,double lambda,
                          double *column_H2, double *column_CO );


// write_Einstein_coeff: write  Einstein A, B or C coefficients
// ------------------------------------------------------------

int write_Einstein_coeff (std::string tag, double *A_coeff, double *B_coeff, double *C_coeff);


// write_R: write transition matrix R
// ----------------------------------

int write_R (std::string tag, long gridp, double *R);


// write_transition_levels: write levels corresponding to each transition
// ----------------------------------------------------------------------

int write_transition_levels (std::string tag, int *irad, int *jrad);


// write_LTE_deviation: write relative deviation of level populations from LTE value
// ---------------------------------------------------------------------------------

// int write_LTE_deviation( std::string tag, CELL *cell, double *energy, double* weight,
//                          double *temperature_gas, double *pop );


// write_true_level_populations: write true level populations
// ----------------------------------------------------------

// int write_true_level_populations( std::string tag, CELL *cell, double *pop );


#endif // __WRITE_TXT_TOOLS_HPP_INCLUDED__
