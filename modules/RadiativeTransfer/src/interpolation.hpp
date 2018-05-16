// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


///  interpolate: interpolate tabulated function for a given range
///  @param[in] *f: pointer to tabulated function values
///  @param[in] *x: pointer to tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return function f evaluated at value
//////////////////////////////////////////////////////////////////

double interpolate (double *f, double *x, long start, long stop, double value);




///  interpolate: interpolate tabulated function for a given range
///  @param[in] *x: pointer to tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return index of x table just above value
//////////////////////////////////////////////////////////////////

long search (double *x, long start, long stop, double value);
