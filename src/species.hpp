// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SPECIES_HPP_INCLUDED__
#define __SPECIES_HPP_INCLUDED__

#include <string>

#include "declarations.hpp"


struct SPECIES
{

  // Chemical symbol

  std::string sym[NSPEC];


  // molecular mass

  double mass[NSPEC];


  // abundance before chemical evolution

  double initial_abundance[NSPEC];


  // species numbers of some inportant species

  int nr_e;      // nr for electrons
  int nr_H2;     // nr for H2
  int nr_HD;     // nr for HD
  int nr_C;      // nr for C
  int nr_H;      // nr for H
  int nr_H2x;    // nr for H2+
  int nr_HCOx;   // nr for HCO+
  int nr_H3x;    // nr for H3+
  int nr_H3Ox;   // nr for H3O+
  int nr_Hex;    // nr for He+
  int nr_CO;     // nr for CO


  // Constructor reads species data file

  SPECIES (std::string spec_datafile);


  // Tools:


  // Get name of species as it appears in species file

  std::string get_canonical_name (std::string name);


  // Get number for given species symbol

  int get_species_nr (std::string name);


  // Check whether it is ortho or para H2

  char check_ortho_para (std::string name);


  // Get charge of a species

  int get_charge (std::string name);


  // Initialize electron abundance so that cell is neutral

  double get_electron_abundance ();

};


#endif // __SPECIES_HPP_INCLUDED__
