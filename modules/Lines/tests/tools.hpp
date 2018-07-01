// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


///  relative_error: compute the relative difference between two values
///    @param[in] a: value 1
///    @paran[in] b: value 2
///    @return relative differencte between a and b
///////////////////////////////////////////////////////////////////////

double relative_error (const double a, const double b)
{
	return fabs((a-b)/(a+b));
}
