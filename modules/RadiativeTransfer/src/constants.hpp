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

const double INVERSE_SQRT_PI = 0.56418958354;   // inverse square root of Pi ( 1/sqrt(PI) )

const double V_TURB_OVER_C_ALL_SQUARED = 100.0 / (CC * CC);        // (v_turb / c)^2  
const double TWO_KB_OVER_MP_C_SQUARED  = 2.0*KB/ (MP * CC * CC);   // 2.0*Kb/Mp*c^2 


const int N_QUADRATURE_POINTS = 7;

// Only allow for odd N_QUADRATURE_POINTS to ensure the line center is taken as frequency
static_assert (N_QUADRATURE_POINTS%2 == 1);

const int NR_LINE_CENTER = N_QUADRATURE_POINTS / 2;


// Roots and weights for Gauss-Hermite quadrature


//const double H_weights[N_QUADRATURE_POINTS] = {0.0112574, 0.222076, 0.533333, 0.222076, 0.0112574};
//const double   H_roots[N_QUADRATURE_POINTS] = {-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856};

const double H_weights[N_QUADRATURE_POINTS] = {0.0005482688559722185, 0.0307571239675865, 0.24012317860501253, 0.45714285714285724, 0.24012317860501253, 0.0307571239675865, 0.0005482688559722185};
const double   H_roots[N_QUADRATURE_POINTS] = {-2.6519613568352334, -1.6735516287674714, -0.8162878828589648, 0.0, 0.8162878828589648, 1.6735516287674714, 2.6519613568352334};


#endif // __MAGRITTE_CONSTANTS_HPP_INCLUDED__
