// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __READ_INPUT_HPP_INCLUDED__
#define __READ_INPUT_HPP_INCLUDED__


#include <string>

#include "declarations.hpp"


// read_txt_input: read input file
// -------------------------------

int read_txt_input (std::string inputfile, long ncells, CELL *cell);


// read__txt_neighbors: read neighbors
// -----------------------------------

int read_txt_neighbors (std::string file_name, long ncells, CELL *cell);


// read_vtu_input: read input file
// -------------------------------

int read_vtu_input (std::string inputfile, long ncells, CELL *cell);


#endif // __READ_INPUT_HPP_INCLUDED__
