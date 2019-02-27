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

inline long Lines ::
    index (
        const long p,
        const int  l,
        const int  k) const
{
  return k + nrad_cum[l] + p * nlines;
}



///  index:
///////////

inline long Lines ::
    index (
        const long p,
        const long line_index) const
{
  return line_index + p * nlines;
}




///  set_emissivity_and_opacity
///    @param[in] p: number of cell
///    @param[in] l: number of line producing species
/////////////////////////////////////////////////////

inline void Lines ::
    set_emissivity_and_opacity (
	      const long p,
        const int  l           )
{

  for (int k = 0; k < linedata[l].nrad; k++)
  {
    const long i = linedata[l].irad[k];
    const long j = linedata[l].jrad[k];

    const long ind = index (p,l,k);

    emissivity[ind] = HH_OVER_FOUR_PI * linedata[l].A[k] * population[p][l](i);

       opacity[ind] = HH_OVER_FOUR_PI * (  population[p][l](j) * linedata[l].Ba[k]
                                         - population[p][l](i) * linedata[l].Bs[k] );
  }


}




///  set_LTE_level_populations
///    @param[in] abundance_lspec: abundance of line species
///    @param[in] temperature: local gas temperature
///    @param[in] p: number of cell
///    @param[in] l: number of line producing species
///////////////////////////////////////////////////////////

inline void Lines ::
    set_LTE_level_populations (
        const double abundance_lspec,
        const double temperature,
	      const long   p,
        const int    l               )
{


  population_tot[p][l] = abundance_lspec;


  // Calculate fractional LTE level populations

  double partition_function = 0.0;

  for (int i = 0; i < linedata[l].nlev; i++)
  {
    population[p][l](i) = 0.5;//linedata[l].weight[i]
                          //* exp( -linedata[l].energy[i] / (KB*temperature) );

    partition_function += population[p][l](i);
  }


  // Rescale (normalize) LTE level populations

  for (int i = 0; i < linedata[l].nlev; i++)
  {
    population[p][l](i) *= population_tot[p][l] / partition_function;
  }


}




inline void Lines ::
    check_for_convergence (
        const long    p,
        const int     l,
        const double  pop_prec,
              double &error_max,
              double &error_mean)
{

  Eigen::VectorXd dpop = population[p][l] - population_prev1[p][l];
  Eigen::VectorXd spop = population[p][l] + population_prev1[p][l];

  const double min_pop = 1.0E-10 * population_tot[p][l];


  for (int i = 0; i < linedata[l].nlev; i++)
  {
    if (population[p][l](i) > min_pop)
    {
      const double relative_change = 2.0 * fabs (dpop(i) / spop(i));

      if (relative_change > pop_prec)
      {
        fraction_not_converged[l] += 1.0 / (ncells*linedata[l].nlev);
      }

      error_mean += relative_change / (ncells*linedata[l].nlev);

      if (relative_change > error_max)
      {
        error_max = relative_change;
      }
    }
  }


}




///  update_using_Ng_acceleration: perform a Ng accelerated iteration step
///    for level populations. All variable names are based on lecture notes
///    by C.P. Dullemond
///////////////////////////////////////////////////////////////////////////

void Lines ::
    update_using_Ng_acceleration ()
{

  for (int l = 0; l < nlspecs; l++)
  {

    VectorXd1 Q1 (ncells, Eigen::VectorXd (linedata[l].nlev));
    VectorXd1 Q2 (ncells, Eigen::VectorXd (linedata[l].nlev));
    VectorXd1 Q3 (ncells, Eigen::VectorXd (linedata[l].nlev));

    VectorXd1 Wt (ncells, Eigen::VectorXd (linedata[l].nlev));


    OMP_PARALLEL_FOR (p, ncells)
    {
      Q1[p] = population[p][l] - 2.0*population_prev1[p][l] + population_prev2[p][l];
      Q2[p] = population[p][l] -     population_prev1[p][l] - population_prev2[p][l] + population_prev3[p][l];
      Q3[p] = population[p][l] -     population_prev1[p][l];

      for (int i = 0; i < linedata[l].nlev; i++)
      {
        if (population[p][l](i) > 0.0)
        {
          Wt[p](i) = 1.0 / population[p][l](i);
        }

        else
        {
          Wt[p](i) = 1.0;
        }
      }

    }


    double A1 = 0.0;
    double A2 = 0.0;

    double B1 = 0.0;
    double B2 = 0.0;

    double C1 = 0.0;
    double C2 = 0.0;


#   pragma omp parallel for reduction( + : A1, A2, B1, B2, C1, C2)
    for (long p = 0; p < ncells; p++)
    {
      A1      = A1 + Q1[p].dot((Wt[p].asDiagonal())*Q1[p]);
      A2 = B1 = A2 + Q1[p].dot((Wt[p].asDiagonal())*Q2[p]);
      B2      = B2 + Q2[p].dot((Wt[p].asDiagonal())*Q2[p]);
      C1      = C1 + Q1[p].dot((Wt[p].asDiagonal())*Q3[p]);
      C2      = C2 + Q2[p].dot((Wt[p].asDiagonal())*Q3[p]);
    }

    const double denominator = A1*B2 - A2*B1;

    if (denominator != 0.0)
    {
      const double a = (C1*B2 - C2*B1) / denominator;
      const double b = (C2*A1 - C1*A2) / denominator;

      OMP_PARALLEL_FOR (p, ncells)
      {
        const Eigen::VectorXd pop_tmp = population[p][l];

        population[p][l] = (1.0 - a - b)*population[p][l]
                                     + a*population_prev1[p][l]
                                     + b*population_prev2[p][l];

        population_prev3[p][l] = population_prev2[p][l];
        population_prev2[p][l] = population_prev1[p][l];
        population_prev1[p][l] = pop_tmp;
      }
    }

  } // end of l loop


}
