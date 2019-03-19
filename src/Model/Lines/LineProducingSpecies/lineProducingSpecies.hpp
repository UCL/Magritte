// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINEPRODUCINGSPECIES_HPP_INCLUDED__
#define __LINEPRODUCINGSPECIES_HPP_INCLUDED__

#include <vector>

#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"
#include "Model/Lines/LineProducingSpecies/Linedata/linedata.hpp"
#include "Model/Lines/LineProducingSpecies/Quadrature/quadrature.hpp"
#include "Model/Lines/LineProducingSpecies/Lambda/lambda.hpp"


struct LineProducingSpecies
{

  public:

      Linedata   linedata;             ///< data for line producing species

      Quadrature quadrature;           ///< data for integral over line

      std::vector<std::vector<Lambda>> lambda;
      Double2 Jeff;
      Double2 Jlin;

      Long3 nr_line;                   ///< frequency number corresponing to line (p,k,z)

      double relative_change_mean;     ///< mean    relative change
      double relative_change_max;      ///< maximum relative change

      double fraction_not_converged;   ///< fraction of levels that is not converged

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

      inline double get_emissivity (
          const long p,
          const long k             ) const;

      inline double get_opacity (
          const long p,
          const long k             ) const;

      inline void check_for_convergence (
          const double pop_prec         );

      inline void update_using_LTE (
          const Double2 &abundance,
          const Double1 &temperature);

      inline void update_using_statistical_equilibrium (
          const Double2 &abundance,
          const Double1 &temperature                   );

      inline void update_using_Ng_acceleration ();


      int initialize_Lambda ();


      long ncells;


  private:

      long nquads;

      static const string prefix;


};


#include "lineProducingSpecies.tpp"


#endif // __LINEPRODUCINGSPECIES_HPP_INCLUDED__
