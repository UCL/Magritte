// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINEPRODUCINGSPECIES_HPP_INCLUDED__
#define __LINEPRODUCINGSPECIES_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"
#include "Model/Lines/LineProducingSpecies/Linedata/linedata.hpp"


struct LineProducingSpecies
{

  public:

      Linedata linedata;               ///< data for line producing species

      Double2 J_line;                  ///< mean intensity in the line
      Double2 J_star;                  ///< approximated mean intensity

      double relative_change_mean;     ///< mean    relative change
      double relative_change_max;      ///< maximum relative change

      double fraction_not_converged;   ///< fraction of levels that is not converged
      bool            not_converged;   ///< true when species is not converged

      VectorXd population;             ///< level population (most recent)
      Double1  population_tot;         ///< total level population (sum over levels)

      VectorXd population_prev1;       ///< level populations 1 iteration  back
      VectorXd population_prev2;       ///< level populations 2 iterations back
      VectorXd population_prev3;       ///< level populations 3 iterations back


      // Io
      int read (
          const Io         &io,
          const long        l,
                Parameters &parameters);

      int write (
          const Io  &io,
          const long l  ) const;


      // Inline functions
      inline long index (
          const long p,
          const long i  ) const;

      inline void set_LTE_level_populations (
          const double abundance_lspec,
          const double temperature,
	        const long   p                    );

      inline void check_for_convergence (
          const double pop_prec         );

      inline void update_using_Ng_acceleration ();


  private:

      long ncells;

      static const string prefix;


};


#include "lineProducingSpecies.tpp"


#endif // __LINEPRODUCINGSPECIES_HPP_INCLUDED__
