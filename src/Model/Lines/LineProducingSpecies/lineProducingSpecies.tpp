// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/constants.hpp"
#include "Tools/types.hpp"
#include "Tools/debug.hpp"


///  index:
///////////

inline long LineProducingSpecies ::
    index (
        const long p,
        const long i ) const
{
  return i + p * linedata.nlev;
}




///  set_LTE_level_populations
///    @param[in] abundance_lspec: abundance of line species
///    @param[in] temperature: local gas temperature
///    @param[in] p: number of cell
///    @param[in] l: number of line producing species
///////////////////////////////////////////////////////////

inline void LineProducingSpecies ::
    set_LTE_level_populations (
        const double abundance_lspec,
        const double temperature,
	      const long   p               )
{


  population_tot[p] = abundance_lspec;


  // Calculate fractional LTE level populations

  double partition_function = 0.0;

  for (int i = 0; i < linedata.nlev; i++)
  {
    const long ind = index (p, i);

    population (ind) = linedata.weight[i]
                       * exp( -linedata.energy[i] / (KB*temperature) );

    partition_function += population (ind);
  }


  // Rescale (normalize) LTE level populations

  for (int i = 0; i < linedata.nlev; i++)
  {
    const long ind = index (p, i);

    population (ind) *= population_tot[p] / partition_function;
  }


}




inline void LineProducingSpecies ::
    check_for_convergence (
        const double pop_prec)
{

  const double weight = 1.0 / (ncells * linedata.nlev);

  fraction_not_converged = 0.0;

  relative_change_mean = 0.0;
  relative_change_max  = 0.0;


  OMP_PARALLEL_FOR (p, ncells)
  {
    const double min_pop = 1.0E-10 * population_tot[p];

    for (int i = 0; i < linedata.nlev; i++)
    {
      const long ind = index (p, i);

      if (population(ind) > min_pop)
      {
        double relative_change = 2.0;

        relative_change *= fabs (population (ind) - population_prev1 (ind));
        relative_change /=      (population (ind) + population_prev1 (ind));

        cout << relative_change << endl;

        if (relative_change > pop_prec)
        {
          fraction_not_converged += weight;
        }


        relative_change_mean += (weight * relative_change);


        if (relative_change > relative_change_max)
        {
          relative_change_max = relative_change;
        }

      }
    }
  }


}




///  update_using_Ng_acceleration: perform a Ng accelerated iteration step
///    for level populations. All variable names are based on lecture notes
///    by C.P. Dullemond
///////////////////////////////////////////////////////////////////////////

void LineProducingSpecies ::
    update_using_Ng_acceleration ()
{

  VectorXd Wt (ncells*linedata.nlev);

  VectorXd Q1 = population - 2.0*population_prev1 + population_prev2;
  VectorXd Q2 = population -     population_prev1 - population_prev2 + population_prev3;
  VectorXd Q3 = population -     population_prev1;


  //OMP_PARALLEL_FOR (ind, ncells*linedata.nlev)
  //{
  //  if (population (ind) > 0.0)
  //  {
  //    Wt (ind) = 1.0 / population (ind);
  //  }

  //  else
  //  {
  //    Wt (ind) = 1.0;
  //  }
  //}


  //const double A1 = Q1.dot (Wt.asDiagonal()*Q1);
  //const double A2 = Q1.dot (Wt.asDiagonal()*Q2);
  //const double B2 = Q2.dot (Wt.asDiagonal()*Q2);
  //const double C1 = Q1.dot (Wt.asDiagonal()*Q3);
  //const double C2 = Q2.dot (Wt.asDiagonal()*Q3);

  const double A1 = Q1.dot(Q1);
  const double A2 = Q1.dot(Q2);
  const double B2 = Q2.dot(Q2);
  const double C1 = Q1.dot(Q3);
  const double C2 = Q2.dot(Q3);

  const double B1 = A2;

  const double denominator = A1*B2 - A2*B1;


  if (denominator != 0.0)
  {
    const VectorXd pop_tmp = population;

    const double a = (C1*B2 - C2*B1) / denominator;
    const double b = (C2*A1 - C1*A2) / denominator;

    population = (1.0 - a - b)*population
                           + a*population_prev1
                           + b*population_prev2;

    population_prev3 = population_prev2;
    population_prev2 = population_prev1;
    population_prev1 = pop_tmp;
  }

}
