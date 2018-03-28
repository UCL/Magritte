// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRITE_TXT_TOOLS_HPP_INCLUDED__
#define __WRITE_TXT_TOOLS_HPP_INCLUDED__


#include <string>

#include "declarations.hpp"


// write_txt_output: write output in txt format
// --------------------------------------------

int write_txt_output (std::string tag, long ncells, CELLS *cells, LINES lines);


// write_grid: write grid
// ----------------------

int write_grid (std::string tag, long ncells, CELLS *cells);


// write_neighbors: write neighbors of each cell
// ---------------------------------------------

int write_neighbors (std::string tag, long ncells, CELLS *cells);


// write_rays: write unit HEALPix vectors
// ------------------------------------------------

int write_rays (std::string tag, RAYS rays);


// write_abundances: write abundances at each point
// ------------------------------------------------

int write_abundances (std::string tag, long ncells, CELLS *cells);


// write_level_populations: write level populations at each point for each transition
// ----------------------------------------------------------------------------------

int write_level_populations (std::string tag, long ncells, CELLS *cells, LINES lines);


// // write_line_intensities: write line intensities for each species, point and transition
// // -------------------------------------------------------------------------------------
//
// int write_line_intensities (std::string tag, long ncells, CELLS *cells, LINES lines);
//

// write_thermal_ratio: write thermal ratio at each cell
// -----------------------------------------------------

int write_thermal_ratio (std::string tag, long ncells, CELLS *cells);


// write_thermal_ratio_prev: write thermal ratio at each cell
// -----------------------------------------------------

int write_thermal_ratio_prev (std::string tag, long ncells, CELLS *cells);


// write_temperature_gas: write gas temperatures at each point
// -----------------------------------------------------------

int write_temperature_gas (std::string tag, long ncells, CELLS *cells);


// write_temperature_dust: write dust temperatures at each point
// -------------------------------------------------------------

int write_temperature_dust (std::string tag, long ncells, CELLS *cells);


// write_temperature_gas_prev: write previous gas temperatures at each point
//--------------------------------------------------------------------------

int write_temperature_gas_prev (std::string tag, long ncells, CELLS *cells);
//
//
// // write_UV_field: write UV field at each point
// // --------------------------------------------
//
// int write_UV_field (std::string tag, long ncells, double *UV_field);
//
//
// // write_UV_field: write UV field at each point
// // --------------------------------------------
//
// int write_AV (std::string tag, long ncells, double *AV);
//
//
// // write_rad_surface: write rad surface at each point
// // --------------------------------------------------
//
// int write_rad_surface (std::string tag, long ncells, double *rad_surface);
//
//
// // write_reaction_rates: write rad surface at each point
// // -----------------------------------------------------
//
// int write_reaction_rates (std::string tag, long ncells, CELLS *cells);
//
//
// // write_certain_reactions: write rates of certain reactions
// // ---------------------------------------------------------
//
// int write_certain_rates (std::string tag, long ncells, CELLS *cells, std::string name,
//                          int nr_certain_reac, int *certain_reactions);


// write_double_vector: write a vecotr of doubles
// ----------------------------------------------

int write_double_vector (std::string name, std::string tag, long length, const double *variable);


// write_double_matrix: write a matrix of doubles
// ----------------------------------------------

int write_double_matrix (std::string name, std::string tag, long nrows, long ncols, const double *variable);

//
// // // write_radfield_tools: write output of functoins defined in radfield_tools
// // // -------------------------------------------------------------------------
// //
// // int write_radfield_tools (std::string tag, double *AV ,double lambda,
// //                           double *column_H2, double *column_CO );
//
//
// // write_Einstein_coeff: write  Einstein A, B or C coefficients
// // ------------------------------------------------------------
//
// int write_Einstein_coeff (std::string tag, LINES lines, double *C_coeff);
//
//
// // write_R: write transition matrix R
// // ----------------------------------
//
// int write_R (std::string tag, long ncells, LINES lines, long o, double *R);
//
//
// // write_transition_levels: write levels corresponding to each transition
// // ----------------------------------------------------------------------
//
// int write_transition_levels (std::string tag, LINES lines);
//
//
// // write_LTE_deviation: write relative deviation of level populations from LTE value
// // ---------------------------------------------------------------------------------
//
// // int write_LTE_deviation( std::string tag, CELLS *cells, double *energy, double* weight,
// //                          double *temperature_gas, double *pop );
//
//
// // write_true_level_populations: write true level populations
// // ----------------------------------------------------------
//
// // int write_true_level_populations( std::string tag, CELLS *cells, double *pop );


#endif // __WRITE_TXT_TOOLS_HPP_INCLUDED__
