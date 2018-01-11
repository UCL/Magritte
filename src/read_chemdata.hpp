// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __READ_CHEMDATA_HPP_INCLUDED__
#define __READ_CHEMDATA_HPP_INCLUDED__


#include <string>


// read_species: read species from data file
// -----------------------------------------

int read_species (std::string spec_datafile, long ncells, CELL *cell, SPECIES *species);


// read_reactions: read reactoins from (CSV) data file
// ---------------------------------------------------

int read_reactions (std::string reac_datafile);


#endif // __READ_CHEMDATA_HPP_INCLUDED__
