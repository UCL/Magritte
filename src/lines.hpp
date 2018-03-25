// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINES_HPP_INCLUDED__
#define __LINES_HPP_INCLUDED__

#include "declarations.hpp"
#include "line_data.hpp"


struct LINES
{

  // int nr[NLSPEC];                // symbol of line producing species
  // std::string sym[NLSPEC];       // symbol of line producing species
  //
  //
  // int irad[TOT_NRAD];            // level index of radiative transition
  // int jrad[TOT_NRAD];            // level index of radiative transition
  //
  // double energy[TOT_NLEV];       // energy of level
  // double weight[TOT_NLEV];       // statistical weight of level
  //
  // double frequency[TOT_NLEV2];   // frequency corresponing to i -> j transition
  //
  // double A_coeff[TOT_NLEV2];     // Einstein A_ij coefficient
  // double B_coeff[TOT_NLEV2];     // Einstein B_ij coefficient
  //
  //
  // // Collision related variables
  //
  // int partner[TOT_NCOLPAR];                  // species number corresponding to a collision partner
  //
  // char ortho_para[TOT_NCOLPAR];              // stores whether it is ortho or para H2
  //
  // double coltemp[TOT_CUM_TOT_NCOLTEMP];      // Collision temperatures for each partner
  //
  // double C_data[TOT_CUM_TOT_NCOLTRANTEMP];   // C_data for each partner, tran. and temp.
  //
  // int icol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition
  // int jcol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition
  //


  int nr[NLSPEC]          = NUMBER;         // symbol of line producing species
  std::string sym[NLSPEC] = NAME;       // symbol of line producing species

  int irad[TOT_NRAD] = IRAD;            // level index of radiative transition
  int jrad[TOT_NRAD] = JRAD;            // level index of radiative transition

  double energy[TOT_NLEV] = ENERGY;       // energy of level
  double weight[TOT_NLEV] = WEIGHT;       // statistical weight of level

  double frequency[TOT_NLEV2] = FREQUENCY;   // frequency corresponing to i -> j transition

  double A_coeff[TOT_NLEV2] = A_COEFF;     // Einstein A_ij coefficient
  double B_coeff[TOT_NLEV2] = B_COEFF;     // Einstein B_ij coefficient


  // Collision related variables

  int partner[TOT_NCOLPAR] = PARTNER_NR;              // species number corresponding to a collision partner

  char ortho_para[TOT_NCOLPAR] = ORTHO_PARA;          // stores whether it is ortho or para H2

  double coltemp[TOT_CUM_TOT_NCOLTEMP] = COLTEMP;     // Collision temperatures for each partner

  double C_data[TOT_CUM_TOT_NCOLTRANTEMP] = C_DATA;   // C_data for each partner, tran. and temp.

  int icol[TOT_CUM_TOT_NCOLTRAN] = ICOL;            // level index corresp. to col. transition
  int jcol[TOT_CUM_TOT_NCOLTRAN] = JCOL;            // level index corresp. to col. transition






  // source: calculate line source function
  //---------------------------------------

  int source (long ncells, CELL *cell, int lspec, double *source);


  // opacity: calculate line opacity
  // -------------------------------

  int opacity (long ncells, CELL *cell, int lspec, double *opacity);


  // profile: calculate line profile function
  // ----------------------------------------

  double profile (long ncells, CELL *cell, double velocity, double freq, double line_freq, long o);

};


#endif // __LINES_HPP_INCLUDED__
