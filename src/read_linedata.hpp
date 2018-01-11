// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __READ_LINEDATA_HPP_INCLUDED__
#define __READ_LINEDATA_HPP_INCLUDED__


#include <string>


// read_linedata: read data files containing line information in LAMBDA/RADEX format
// ---------------------------------------------------------------------------------

int read_linedata (const std::string *line_datafile, SPECIES *species, int *irad, int *jrad,
                   double *energy, double *weight, double *frequency, double *A_coeff,
                   double *B_coeff, double *coltemp, double *C_data, int *icol, int *jcol);


// extract_spec_par: extract species corresponding to collision partner
// --------------------------------------------------------------------

int extract_spec_par (SPECIES *species, char *buffer, int lspec, int par);


#endif // __READ_LINEDATA_HPP_INCLUDED__
