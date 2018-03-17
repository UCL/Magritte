// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __REACTIONS_HPP_INCLUDED__
#define __REACTIONS_HPP_INCLUDED__

#include <stdio.h>
#include <math.h>

#include "declarations.hpp"


struct REACTIONS
{

  // Reactant symbols

  std::string R1[NREAC];   // reactant 1
  std::string R2[NREAC];   // reactant 2
  std::string R3[NREAC];   // reactant 3


  // Reaction product symbols

  std::string P1[NREAC];   // reaction product 1
  std::string P2[NREAC];   // reaction product 2
  std::string P3[NREAC];   // reaction product 3
  std::string P4[NREAC];   // reaction product 4


  // Reaction coefficients to calculate reaction rate

  double alpha[NREAC];
  double beta[NREAC];
  double gamma[NREAC];
  double RT_min[NREAC];
  double RT_max[NREAC];


  // Number of duplicates of this reaction

  int dup[NREAC];


  // Reaction numbers of some important reactions

  int nr_C_ionization;
  int nr_H2_formation;
  int nr_H2_photodissociation;


  // Constructor reads reaction data file

  REACTIONS (std::string reac_datafile);


  // Checks whether there data closer to actual temperature

  bool no_better_data (int reac, double temperature_gas);

};


#endif // __REACTIONS_HPP_INCLUDED__
