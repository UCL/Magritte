// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_CONSTANTS_HPP_INCLUDED__
#define __MAGRITTE_CONSTANTS_HPP_INCLUDED__


// Numerical constants

#define PI    3.141592653589793238462643383279502884197   // pi
#define CC    2.99792458E+10                              // speed of light in cgs units
#define HH    6.62606896E-27                              // Planck's constant in cgs units
#define KB    1.38065040E-16                              // Boltzmann's constant in cgs units
#define EV    1.60217646E-12                              // one electron Volt in erg
#define MP    1.67262164E-24                              // proton mass in cgs units
#define PC    3.08568025E+18                              // one parsec in cm
#define AU    1.66053878E-24                              // atomic mass unit
#define T_CMB 2.725                                       // CMB  temperature in K

#define SECONDS_IN_YEAR 3.1556926E+7                      // number of seconds in one year


// Roots of 4th (physicists') Hermite polynomial

#define WEIGHTS_4 {0.0458759, 0.0458759, 0.454124, 0.454124}
#define ROOTS_4   {-1.6506801238857844, 1.6506801238857844, -0.5246476232752905, 0.5246476232752905}


// Roots of 5th (physicists') Hermite polynomial

#define WEIGHTS_5 {0.533333, 0.0112574, 0.0112574, 0.222076, 0.222076}
#define ROOTS_5   {0.0, -2.0201828704560856, 2.0201828704560856, -0.9585724646138185, 0.9585724646138185}


// Roots of 7th (physicists') Hermite polynomia

#define WEIGHTS_7 {0.45714285714285724, 0.0005482688559722185, 0.0005482688559722185, 0.24012317860501253, 0.24012317860501253, 0.0307571239675865, 0.0307571239675865}
#define ROOTS_7   {0.0, -2.6519613568352334, 2.6519613568352334, 0.8162878828589648 , -0.8162878828589648, -1.6735516287674714, 1.6735516287674714}


#endif // __MAGRITTE_CONSTANTS_HPP_INCLUDED__
