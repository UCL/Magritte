// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SETUP_HEALPIXVECTORS_HPP_INCLUDED__
#define __SETUP_HEALPIXVECTORS_HPP_INCLUDED__



// create_healpixvector: store HEALPix vectors and find antipodal pairs
// --------------------------------------------------------------------

int setup_healpixvectors (long nrays, double *healpixvector, long *antipod, long *n_aligned, long **aligned, long *mirror_xz);


#endif // __SETUP_HEALPIXVECTORS_HPP_INCLUDED__
