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

const double SECONDS_IN_YEAR = 3.1556926E+7;   // number of seconds in one year


// Roots and weights for Gauss-Hermite quadrature

const double H_4_weights[4] = {0.0458759, 0.0458759, 0.454124, 0.454124};
const double H_4_roots[4]   = {-1.6506801238857844, 1.6506801238857844, -0.5246476232752905, 0.5246476232752905};

const double H_5_weights[5] = {0.533333, 0.0112574, 0.0112574, 0.222076, 0.222076};
const double H_5_roots[5]   = {0.0, -2.0201828704560856, 2.0201828704560856, -0.9585724646138185, 0.9585724646138185};

const double H_7_weights[7] = {0.45714285714285724, 0.0005482688559722185, 0.0005482688559722185, 0.24012317860501253, 0.24012317860501253, 0.0307571239675865, 0.0307571239675865};
const double H_7_roots[7]   = {0.0, -2.6519613568352334, 2.6519613568352334, 0.8162878828589648 , -0.8162878828589648, -1.6735516287674714, 1.6735516287674714};


#endif // __MAGRITTE_CONSTANTS_HPP_INCLUDED__
