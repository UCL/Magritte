// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CONSTANTS_HPP_INCLUDED__
#define __CONSTANTS_HPP_INCLUDED__


// NOTE: We only use SI units thoughout the code.
/////////////////////////////////////////////////


// Numerical constants

const double PI              = 3.141592653589;    // pi
const double FOUR_PI         = 12.56637061436;    // 4*pi
const double INVERSE_SQRT_PI = 0.5641895835478;   // 1/sqrt(pi)

const double CC    = 2.99792458E+8;    // [m/s] speed of light
const double HH    = 6.62607004E-34;   // [J*s] Planck's constant
const double KB    = 1.38064852E-23;   // [J/K] Boltzmann's constant
const double AMU   = 1.66053904E-27;    // [kg] proton mass
const double T_CMB = 2.7254800;        // [K] CMB temperature
//const double EV    = 1.60217646E-12;   // one electron Volt in erg
//const double PC    = 3.08568025E+18;   // one parsec in cm
//const double AU    = 1.66053878E-24;   // atomic mass unit

const double SECONDS_IN_YEAR = 3.1556926E+7;    // number of seconds in one year

const double C_SQUARED = 8.98755179E+16;   // [m^2/s^2] speed of light squared

const double V_TURB_OVER_C_ALL_SQUARED = 0.15E3 * 0.15E3 / C_SQUARED;   // (v_turb / c)^2

//extern const double v_turb;
//extern const double V_TURB_OVER_C_ALL_SQUARED;
//
//#ifndef __VTURB_DEFINED__
//const double v_turb = 10;
//const double V_TURB_OVER_C_ALL_SQUARED = v_turb * v_turb / C_SQUARED;   // (v_turb / c)^2
//#endif

const double TWO_KB_OVER_AMU_C_SQUARED  = 2.0 * KB / (AMU * C_SQUARED);   // 2.0*Kb/(AMU*c^2)

const double TWO_HH_OVER_CC_SQUARED = 2.0 * HH / C_SQUARED;

const double HH_OVER_KB = HH / KB;

const double HH_OVER_FOUR_PI = HH / FOUR_PI;


//const int N_QUADRATURE_POINTS = 1;
//const int N_QUADRATURE_POINTS = 11;
//const int N_QUADRATURE_POINTS = 39;

//// Only allow for odd N_QUADRATURE_POINTS to ensure the line center is taken as frequency
//static_assert (N_QUADRATURE_POINTS%2 == 1, "Number of quadrature points should be odd!");
//
//const int NR_LINE_CENTER = N_QUADRATURE_POINTS / 2;
//
//
//const double LOWER = 1.00001*H_roots[0];
//const double UPPER = 1.00001*H_roots[N_QUADRATURE_POINTS-1];


#endif // __CONSTANTS_HPP_INCLUDED__
