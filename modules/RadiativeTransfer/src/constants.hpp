// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_CONSTANTS_HPP_INCLUDED__
#define __MAGRITTE_CONSTANTS_HPP_INCLUDED__


// Numerical constants

const double PI    = 3.141592653589;   // pi
const double CC    = 2.99792458E+10;   // speed of light in cgs units

const double HH    = 6.62606896E-27;   // Planck's constant in cgs units
const double KB    = 1.38065040E-16;   // Boltzmann's constant in cgs units
const double EV    = 1.60217646E-12;   // one electron Volt in erg
const double MP    = 1.67262164E-24;   // proton mass in cgs units
const double PC    = 3.08568025E+18;   // one parsec in cm
const double AU    = 1.66053878E-24;   // atomic mass unit
const double T_CMB = 2.725;            // CMB  temperature in K

const double SECONDS_IN_YEAR = 3.1556926E+7;    // number of seconds in one year

const double C_SQUARED = 8.98755179E+20;   // speed of light squared in cgs units

const double INVERSE_SQRT_PI = 0.56418958354;   // inverse square root of Pi ( 1/sqrt(PI) )

const double v_turb = 0.12012E5;
const double V_TURB_OVER_C_ALL_SQUARED = v_turb * v_turb / C_SQUARED;   // (v_turb / c)^2

//extern const double v_turb;
//extern const double V_TURB_OVER_C_ALL_SQUARED;
//
//#ifndef __VTURB_DEFINED__
//const double v_turb = 10;
//const double V_TURB_OVER_C_ALL_SQUARED = v_turb * v_turb / C_SQUARED;   // (v_turb / c)^2
//#endif

const double TWO_KB_OVER_MP_C_SQUARED  = 2.0 * KB / (MP * C_SQUARED);   // 2.0*Kb/Mp*c^2

const double TWO_HH_OVER_CC_SQUARED = 2.0 * HH / C_SQUARED;

const double HH_OVER_KB = HH / KB;

const int N_QUADRATURE_POINTS = 11;

// Only allow for odd N_QUADRATURE_POINTS to ensure the line center is taken as frequency
static_assert (N_QUADRATURE_POINTS%2 == 1, "Number of quadrature points should be odd!");

const int NR_LINE_CENTER = N_QUADRATURE_POINTS / 2;


// Roots and weights for Gauss-Hermite quadrature


//const double H_weights[N_QUADRATURE_POINTS] = {0.0112574, 0.222076, 0.533333, 0.222076, 0.0112574};
//const double   H_roots[N_QUADRATURE_POINTS] = {-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856};

//const double H_weights[N_QUADRATURE_POINTS] = {0.0005482688559722185, 0.0307571239675865, 0.24012317860501253, 0.45714285714285724, 0.24012317860501253, 0.0307571239675865, 0.0005482688559722185};
//const double   H_roots[N_QUADRATURE_POINTS] = {-2.6519613568352334, -1.6735516287674714, -0.8162878828589648, 0.0, 0.8162878828589648, 1.6735516287674714, 2.6519613568352334};

const double H_weights[N_QUADRATURE_POINTS] = {8.121849790214925E-7, 0.00019567193027122308, 0.006720285235537279, 0.06613874607105787, 0.24224029987397, 0.3694083694083695, 0.24224029987397, 0.06613874607105787, 0.006720285235537279, 0.00019567193027122308, 8.121849790214925E-7};
const double   H_roots[N_QUADRATURE_POINTS] = {-3.6684708465595826, -2.783290099781652,  -2.0259480158257555, -1.3265570844949328, -0.6568095668820998, 0.0, 0.6568095668820998,  1.3265570844949328, 2.0259480158257555, 2.783290099781652, 3.6684708465595826};

#endif // __MAGRITTE_CONSTANTS_HPP_INCLUDED__
