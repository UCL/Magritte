// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADFIELD_TOOLS_HPP_INCLUDED__
#define __RADFIELD_TOOLS_HPP_INCLUDED__


// self_shielding_H2: Returns H2 self-shielding function
// -----------------------------------------------------

double self_shielding_H2 (double column_H2, double doppler_width, double radiation_width);


// self_shielding_CO: Returns CO self-shielding function
// -----------------------------------------------------

double self_shielding_CO (double column_CO, double column_H2);


// dust_scattering: Retuns attenuation due to scattering by dust
// -------------------------------------------------------------

double dust_scattering (double AV_ray, double lambda);


// X_lambda: Retuns ratio of optical depths at given lambda w.r.t. visual wavelenght
// ---------------------------------------------------------------------------------

double X_lambda (double lambda);


#endif // __RADFIELD_TOOLS_HPP_INCLUDED__
