// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRITE_OUTPUT_HPP_INCLUDED__
#define __WRITE_OUTPUT_HPP_INCLUDED__


#include <string>

#include "declarations.hpp"


// write_txt_output: write output in txt format
// --------------------------------------------

int write_output (long ncells, CELL *cell, LINES lines);


// write_output_log: write output info
// -----------------------------------

int write_output_log ();


// write_performance_log: write performance results
// ------------------------------------------------

int write_performance_log (TIMERS timers);


#endif // __WRITE_OUTPUT_HPP_INCLUDED__
