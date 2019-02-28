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
#include "Model/Lines/Quadrature/quadrature.hpp"


struct Lines
{

  public:

      std::vector <LineProducingSpecies> lineProducingSpecies;


      Quadrature quadrature;


      Double1 line;         ///< [Hz] line center frequencies orderd
      Long1   line_index;   ///< index of the corresponding frequency in line

      Long4 nr_line;        ///< frequency number corresponing to line (p,l,k,z)



      Double1 emissivity;   ///< line emissivity (p,l,k)
      Double1 opacity;      ///< line opacity    (p,l,k)


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


      int gather_emissivities_and_opacities ();


      // Inline functions
      inline long index (
          const long p,
          const int  l,
          const int  k) const;

      inline long index (
          const long p,
          const long line_index) const;

      inline void set_emissivity_and_opacity (
      	  const long p,
          const int  l                       );


  private:

      long ncells;
      long nlines;
      long nlspecs;
      long nquads;

      Long1 nrad_cum;

      static const string prefix;

};


#include "lines.tpp"


#endif // __LINES_HPP_INCLUDED__
