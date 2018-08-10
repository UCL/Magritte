// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_CONSTANTS_HPP_INCLUDED__
#define __MAGRITTE_CONSTANTS_HPP_INCLUDED__


// NOTE: We only use SI units thoughout the code.
/////////////////////////////////////////////////


// Numerical constants

const double PI              = 3.141592653589;    // pi
const double FOUR_PI         = 12.56637061436;    // 4*pi
const double INVERSE_SQRT_PI = 0.5641895835478;   // 1/sqrt(pi)

const double CC    = 2.99792458E+8;    // [m/s] speed of light
const double HH    = 6.62607004E-34;   // [J*s] Planck's constant
const double KB    = 1.38064852E-23;   // [J/K] Boltzmann's constant
const double MP    = 1.6726219E-27;    // [kg] proton mass
const double T_CMB = 2.72548;          // [K] CMB temperature
//const double EV    = 1.60217646E-12;   // one electron Volt in erg
//const double PC    = 3.08568025E+18;   // one parsec in cm
//const double AU    = 1.66053878E-24;   // atomic mass unit

const double SECONDS_IN_YEAR = 3.1556926E+7;    // number of seconds in one year

const double C_SQUARED = 8.98755179E+16;   // [m^2/s^2] speed of light squared


const double v_turb = 0.12012E3;
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

const double HH_OVER_FOUR_PI = HH / FOUR_PI;

const int N_QUADRATURE_POINTS = 11;

// Only allow for odd N_QUADRATURE_POINTS to ensure the line center is taken as frequency
static_assert (N_QUADRATURE_POINTS%2 == 1, "Number of quadrature points should be odd!");

const int NR_LINE_CENTER = N_QUADRATURE_POINTS / 2;


// Roots and weights for Gauss-Hermite quadrature


//const double H_weights[N_QUADRATURE_POINTS] = {0.0112574, 0.222076, 0.533333, 0.222076, 0.0112574};
//const double   H_roots[N_QUADRATURE_POINTS] = {-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856};

//const double H_weights[N_QUADRATURE_POINTS] = {0.0005482688559722185, 0.0307571239675865, 0.24012317860501253, 0.45714285714285724, 0.24012317860501253, 0.0307571239675865, 0.0005482688559722185};
//const double   H_roots[N_QUADRATURE_POINTS] = {-2.6519613568352334, -1.6735516287674714, -0.8162878828589648, 0.0, 0.8162878828589648, 1.6735516287674714, 2.6519613568352334};

const double H_weights[N_QUADRATURE_POINTS] = { 8.121849790214925E-7,
                                                1.956719302712231E-4,
                                                6.720285235537279E-3,
                                                6.613874607105787E-2,
                                                0.24224029987397,
                                                0.36940836940837,
                                                0.24224029987397,
                                                6.613874607105787E-2,
                                                6.720285235537279E-3,
                                                1.956719302712231E-4,
                                                8.121849790214925E-7 };

const double   H_roots[N_QUADRATURE_POINTS] = { -3.668470846559583,
                                                -2.783290099781652,
                                                -2.025948015825756,
                                                -1.326557084494933,
                                                -0.656809566882100,
                                                 0.0,
                                                 0.656809566882100,
                                                 1.326557084494933,
                                                 2.025948015825756,
                                                 2.783290099781652,
                                                 3.668470846559583 };

#endif // __MAGRITTE_CONSTANTS_HPP_INCLUDED__
