// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SPECIES_HPP_INCLUDED__
#define __SPECIES_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


struct Species
{

  public:

      String1 sym;                 ///< chemical symbol of species
      //Double1 mass;                ///< molecular mass of species

      Double2 abundance_init;      ///< abundance before chemical evolution
      Double2 abundance;           ///< (current) abundance in every cell


      // species numbers of some inportant species
      //long nr_e;      // nr for electrons
      //long nr_H2;     // nr for H2
      //long nr_HD;     // nr for HD
      //long nr_C;      // nr for C
      //long nr_H;      // nr for H
      //long nr_H2x;    // nr for H2+
      //long nr_HCOx;   // nr for HCO+
      //long nr_H3x;    // nr for H3+
      //long nr_H3Ox;   // nr for H3O+
      //long nr_Hex;    // nr for He+
      //long nr_CO;     // nr for CO


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


    // Helper function
    long get_species_nr (
        const string name) const;


  private:

      long ncells;   ///< number of cells
      long nspecs;   ///< number of chemical species

      static const string prefix;


};


#endif // __SPECIES_HPP_INCLUDED__
