// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SPECIES_HPP_INCLUDED__
#define __SPECIES_HPP_INCLUDED__

#include <string>
#include <vector>
using namespace std;


struct SPECIES
{

	long ncells;                        ///< total number of cells

	int nspec;                          ///< total number of chemical species

  vector<string> sym;                 ///< symbol of chemical species

  vector<double> mass;                ///< molecular mass of species

  vector<double> initial_abundance;   ///< abundance before chemical evolution

	vector<vector<double>> abundance;   ///< abundance for every cell


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


  // Constructor: reads species data file
  // ------------------------------------

  SPECIES (long num_of_cells, int num_of_spec, string spec_datafile);


  int read (string file_name);


  // Tools:


  // get_canonical_name: Get name of species as it appears in species file
  // ---------------------------------------------------------------------

  string get_canonical_name (string name);


  // get_species_nr: Get number for given species symbol
  // ---------------------------------------------------

  int get_species_nr (string name);


  // check_ortho_para: Check whether it is ortho or para H2
  // ------------------------------------------------------

  char check_ortho_para (string name);


  // get_charge: Get charge of a species
  // -----------------------------------

  int get_charge (string name);


  // get_electron_abundance: Initialize electron abundance so that cell is neutral
  // -----------------------------------------------------------------------------

  double get_electron_abundance ();

};


#endif // __SPECIES_HPP_INCLUDED__
