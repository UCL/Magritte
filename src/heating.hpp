// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __HEATING_HPP_INCLUDED__
#define __HEATING_HPP_INCLUDED__

#include "declarations.hpp"


// heating: calculate total heating
// --------------------------------

double heating (long ncells, CELL *cell, long gridp, double* heating_components);


// F: mathematical function used in photoelectric dust heating
// -----------------------------------------------------------

double F (double x, double delta, double gamma);


// dF: defivative w.r.t. x of function F defined above
// ---------------------------------------------------

double dF (double x, double delta);


#endif // __HEATING_HPP_INCLUDED__
