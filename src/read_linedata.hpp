// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __READ_LINEDATA_HPP_INCLUDED__
#define __READ_LINEDATA_HPP_INCLUDED__


#include <string>


// read_linedata: read data files containing line information in LAMBDA/RADEX format
// ---------------------------------------------------------------------------------

int read_linedata (const std::string *line_datafile, LINE_SPECIES *line_species, SPECIES *species);


// extract_collision_partner: extract species corresponding to collision partner
// -----------------------------------------------------------------------------

int extract_collision_partner (SPECIES *species, LINE_SPECIES *line_species, char *buffer, int lspec, int par);


#endif // __READ_LINEDATA_HPP_INCLUDED__
