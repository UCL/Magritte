// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINES_HPP_INCLUDED__
#define __LINES_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"
#include "Model/Lines/LineProducingSpecies/lineProducingSpecies.hpp"


struct Lines
{

  public:

      std::vector <LineProducingSpecies> lineProducingSpecies;

      Double1 line;         ///< [Hz] line center frequencies orderd
      Long1   line_index;   ///< index of the corresponding frequency in line

      Double1 emissivity;   ///< line emissivity (p,l,k)
      Double1 opacity;      ///< line opacity    (p,l,k)


      // Io
      void read  (const Io &io, Parameters &parameters);
      void setup (              Parameters &parameters);
      void write (const Io &io                        ) const;


      int iteration_using_LTE (
          const Double2 &abundance,
          const Double1 &temperature);

      int iteration_using_statistical_equilibrium (
          const Double2 &abundance,
          const Double1 &temperature,
          const double   pop_prec                 );

      int iteration_using_Ng_acceleration (
          const double   pop_prec         );


      // Inline functions
      inline long index (
          const long p,
          const int  l,
          const int  k) const;

      inline long index (
          const long p,
          const long line_index) const;

      inline void set_emissivity_and_opacity ();


      int gather_emissivities_and_opacities ();


  private:

      size_t ncells;
      size_t nlines;
      size_t nlspecs;

      Long1 nrad_cum;

      static const string prefix;

};


#include "lines.tpp"


#endif // __LINES_HPP_INCLUDED__
