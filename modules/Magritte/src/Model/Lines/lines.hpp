// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINES_HPP_INCLUDED__
#define __LINES_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"
#include "Model/Lines/Linedata/linedata.hpp"


struct Lines
{

  public:

      // Data

      std::vector <Linedata> linedata;   ///< data for each line producing species

      Double1 quadrature_roots;
      Double1 quadrature_weights;


      // Lines

      Double1 line;                       ///< [Hz] line center frequencies orderd
      Long1   line_index;                 ///< index of the corresponding frequency in line

      Long4 nr_line;                      ///< frequency number corresponing to line (p,l,k,z)

      Double3 J_line;                     ///< mean intensity in the line
      Double3 J_star;                     ///< approximated mean intensity

      Double1 emissivity;   ///< line emissivity (p,l,k)
      Double1 opacity;      ///< line opacity    (p,l,k)


      // Levels

      Bool1            not_converged;    ///< true when species is not converged
      Double1 fraction_not_converged;    ///< fraction of levels that is not converged

      VectorXd2 population;              ///< level population (most recent)
      Double2   population_tot;          ///< total level population (sum over levels)

      VectorXd2 population_prev1;        ///< level populations 1 iteration  back
      VectorXd2 population_prev2;        ///< level populations 2 iterations back
      VectorXd2 population_prev3;        ///< level populations 3 iterations back


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

      inline void set_LTE_level_populations (
          const double abundance_lspec,
          const double temperature,
	        const long   p,
          const int    l                    );

      inline void set_emissivity_and_opacity (
      	  const long p,
          const int  l                       );

      inline void check_for_convergence (
          const long    p,
          const int     l,
          const double  pop_prec,
                double &error_max,
                double &error_mean      );

      inline void update_using_Ng_acceleration ();


  private:

      long ncells;
      long nlines;
      long nquads;       ///< number frequency quadrature points
      long nlspecs;

      Long1 nrad_cum;

      static const string prefix;

};


#include "lines.tpp"


#endif // __LINES_HPP_INCLUDED__
