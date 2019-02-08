// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SPECIES_HPP_INCLUDED__
#define __SPECIES_HPP_INCLUDED__

#include "types.hpp"
#include "io.hpp"


struct Species
{

  public:

      long ncells;                 ///< number of cells
      long nspecs;                 ///< number of chemical species

      String1 sym;                 ///< chemical symbol of species
      Double1 mass;                ///< molecular mass of species
      Double1 initial_abundance;   ///< abundance before chemical evolution

      Double2 abundance;           ///< (current) abundance in every cell


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


      // Io
      int read (
          const Io &);

      int write (
          const Io &) const;


  private:

      // Helper functions
      int get_species_nr (
          const string name);


};


#endif // __SPECIES_HPP_INCLUDED__
