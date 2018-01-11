// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SPECIES_TOOLS_HPP_INCLUDED__
#define __SPECIES_TOOLS_HPP_INCLUDED__

#include <string>


// get_canonical_name: get name of species as it appears in species.dat file
// -------------------------------------------------------------------------

std::string get_canonical_name (std::string name);


// get_species_nr: get number corresponding to given species symbol
// ----------------------------------------------------------------

int get_species_nr (SPECIES *species, std::string name);


// check_ortho_para: check whether it is ortho or para H2
// ------------------------------------------------------

char check_ortho_para (std::string name);


// get_charge: get charge of a species
// -----------------------------------

int get_charge (std::string name);


// get_electron_abundance: initialize electron abundance so that gas is neutral
// ----------------------------------------------------------------------------

double get_electron_abundance (long ncells, CELL *cell, SPECIES *species, long gridp);


// no_better_data: checks whether there data closer to actual temperature
// ----------------------------------------------------------------------

bool no_better_data (int reac, REACTION *reaction, double temperature_gas);


#endif // __SPECIES_TOOLS_HPP_INCLUDED__
