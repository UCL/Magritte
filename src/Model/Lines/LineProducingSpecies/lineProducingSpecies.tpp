// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <Eigen/SparseLU>
using Eigen::SparseLU;
#include <Eigen/SparseCore>
using Eigen::SparseMatrix;
using Eigen::SparseVector;
using Eigen::Triplet;
using Eigen::COLAMDOrdering;

//#include <Eigen/IterativeLinearSolvers>
//#include <Eigen/SparseCholesky>

#include <Eigen/Core>

#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/constants.hpp"
#include "Tools/types.hpp"
#include "Tools/logger.hpp"


///  Indexer for level populations
///    @param[in] p : index of the cell
///    @param[in] i : index of the level
///    @return corresponding index for p and i
//////////////////////////////////////////////

inline long LineProducingSpecies ::
    index (
        const long p,
        const long i ) const
{
  return i + p * linedata.nlev;
}




///  Getter for the line emissivity
///    @param[in] p : index of the cell
///    @param[in] k : index of the transition
///    @return line emissivity for cell p and transition k
//////////////////////////////////////////////////////////

inline double LineProducingSpecies ::
    get_emissivity (
        const long p,
        const long k ) const
{
  const long ind_i = index (p, linedata.irad[k]);

  return HH_OVER_FOUR_PI * linedata.A[k] * population(ind_i);
}




///  Getter for the line opacity
///    @param[in] p : index of the cell
///    @param[in] k : index of the transition
///    @return line opacity for cell p and transition k
///////////////////////////////////////////////////////

inline double LineProducingSpecies ::
    get_opacity (
        const long p,
        const long k ) const
{
  const long ind_i = index (p, linedata.irad[k]);
  const long ind_j = index (p, linedata.jrad[k]);

  return HH_OVER_FOUR_PI * (  population(ind_j) * linedata.Ba[k]
                            - population(ind_i) * linedata.Bs[k] );
}




///  set_LTE_level_populations
///    @param[in] abundance_lspec: abundance of line species
///    @param[in] temperature: local gas temperature
///    @param[in] p: number of cell
///    @param[in] l: number of line producing species
///////////////////////////////////////////////////////////

inline void LineProducingSpecies ::
    update_using_LTE (
        const Double2 &abundance,
        const Double1 &temperature)
{


  OMP_PARALLEL_FOR (p, ncells)
  {
    population_tot[p] = abundance[p][linedata.num];

    double partition_function = 0.0;

    for (int i = 0; i < linedata.nlev; i++)
    {
      const long ind = index (p, i);

      population(ind) = linedata.weight[i]
                        * exp (-linedata.energy[i] / (KB*temperature[p]));

      partition_function += population (ind);
    }

    for (int i = 0; i < linedata.nlev; i++)
    {
      const long ind = index (p, i);

      population(ind) *= population_tot[p] / partition_function;
      //population(ind) *= 1.0 / partition_function;
    }
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
///    by C.P. Dullemond which are based on Olson, Auer and Buchler (1985).
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
  //    Wt (ind) = Jlin[p][k];
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




///  update_using_statistical_equilibrium: computes level populations by solving
///  the statistical equilibrium equation taking into account the radiation field
///    @param[in] abundance: chemical abundances of species in the model
///    @param[in] temperature: gas temperature in the model
/////////////////////////////////////////////////////////////////////////////////

inline void LineProducingSpecies ::
    update_using_statistical_equilibrium (
        const Double2 &abundance,
        const Double1 &temperature       )
{

  const long non_zeros = ncells * (    linedata.nlev
                                   + 6*linedata.nrad
                                   + 4*linedata.ncol_tot );


  population_prev3 = population_prev2;
  population_prev2 = population_prev1;
  population_prev1 = population;


  SparseMatrix<double> RT (ncells*linedata.nlev, ncells*linedata.nlev);

  VectorXd y = VectorXd::Zero (ncells*linedata.nlev);

  vector<Triplet<double, int>> triplets;

  triplets.reserve (non_zeros);

  // !!! push_back is not thread safe !!!

  //OMP_PARALLEL_FOR (p, ncells)
  for (long p = 0; p < ncells; p++)
  {

    // Radiative transitions

    for (long k = 0; k < linedata.nrad; k++)
    {
      //cout<<"Jeff["<<p<<"]["<<k<<"] = "<<Jeff[p][k]<<endl;
      //cout<<"Jlin["<<p<<"]["<<k<<"] = "<<Jlin[p][k]<<endl;

      const double v_IJ = linedata.A[k] + linedata.Bs[k] * Jeff[p][k];
      const double v_JI =                 linedata.Ba[k] * Jeff[p][k];


      // Note: we define our transition matrix as the transpose of R in the paper.

      const long I = index (p, linedata.irad[k]);
      const long J = index (p, linedata.jrad[k]);

      if (linedata.jrad[k] != linedata.nlev-1)
      {
        triplets.push_back (Triplet<double, int> (J, I, +v_IJ));
        triplets.push_back (Triplet<double, int> (J, J, -v_JI));
        //triplets.push_back (Eigen::Triplet<double, int> (J, J, -v_IJ));
      }

      if (linedata.irad[k] != linedata.nlev-1)
      {
        triplets.push_back (Triplet<double, int> (I, J, +v_JI));
        triplets.push_back (Triplet<double, int> (I, I, -v_IJ));
        //triplets.push_back (Eigen::Triplet<double, int> (I, I, -v_JI));
      }
    }


    // Approximated Lambda operator

    for (long k = 0; k < linedata.nrad; k++)
    {
      for (long m = 0; m < lambda[p][k].nr.size(); m++)
      {
        const double v_IJ = -get_opacity (p, k) * lambda[p][k].Ls[m];

        // Note: we define our transition matrix as the transpose of R in the paper.

        const long I = index (lambda[p][k].nr[m], linedata.irad[k]);
        const long J = index (p,                  linedata.jrad[k]);

        if (linedata.jrad[k] != linedata.nlev-1)
        {
          triplets.push_back (Triplet<double, int> (J, I, +v_IJ));
        }

        if (linedata.irad[k] != linedata.nlev-1)
        {
          triplets.push_back (Triplet<double, int> (I, I, -v_IJ));
        }
      }
    }



    // Collisional transitions

    for (CollisionPartner &colpar : linedata.colpar)
    {
      double abn = abundance[p][colpar.num_col_partner];
      double tmp = temperature[p];

      colpar.adjust_abundance_for_ortho_or_para (tmp, abn);
      colpar.interpolate_collision_coefficients (tmp);

      //cout << "Is it here?" << endl;
      // Moved interpolation for excitation rate here...
      //for (long k = 0; k < colpar.ncol; k++)
      //{
      //  //cout << "k = " << k << endl;
      //  const long i = colpar.icol[k];
      //  const long j = colpar.jcol[k];

      //  colpar.Ce_intpld[k] = colpar.Cd_intpld[k] * linedata.weight[i] / linedata.weight[j] * exp ( - HH*linedata.frequency[k] / (KB*tmp) );
      //}
      //
      //cout << "Nope..." << endl;

      for (long k = 0; k < colpar.ncol; k++)
      {
        const double v_IJ = colpar.Cd_intpld[k] * abn;
        const double v_JI = colpar.Ce_intpld[k] * abn;


        // Note: we define our transition matrix as the transpose of R in the paper.

        const long I = index (p, colpar.icol[k]);
        const long J = index (p, colpar.jcol[k]);

        if (colpar.jcol[k] != linedata.nlev-1)
        {
          triplets.push_back (Triplet<double, int> (J, I, +v_IJ));
          triplets.push_back (Triplet<double, int> (J, J, -v_JI));
          //triplets.push_back (Eigen::Triplet<double, int> (J, J, -v_IJ));
        }

        if (colpar.icol[k] != linedata.nlev-1)
        {
          triplets.push_back (Triplet<double, int> (I, J, +v_JI));
          triplets.push_back (Triplet<double, int> (I, I, -v_IJ));
          //triplets.push_back (Eigen::Triplet<double, int> (I, I, -v_JI));
        }
      }
    }


    for (long i = 0; i < linedata.nlev; i++)
    {
      const long I = index (p, linedata.nlev-1);
      const long J = index (p, i);

      triplets.push_back (Triplet<double, int> (I, J, 1.0));
    }

    //y.insert (index (p, linedata.nlev-1)) = population_tot[p];
    y[index (p, linedata.nlev-1)] = population_tot[p];
    //y.insert (index (p, linedata.nlev-1)) = 1.0;//population_tot[p];

  } // for all cells

  cout << omp_get_num_threads() << endl;


  RT.setFromTriplets (triplets.begin(), triplets.end());


  //cout << "Compressing RT" << endl;

  //RT.makeCompressed ();


  //Eigen::BiCGSTAB <SparseMatrix<double>> solver;

  //cout << "Try compute" << endl;

  //solver.compute (RT);

  //if (solver.info() != Eigen::Success)
  //{
  //  cout << "Decomposition failed" << endl;
  //  //assert(false);
  //}


  //for (int tel=0; tel<5; tel++)
  //{
  //  //Eigen::Gues x0 = population;

  //  population = solver.solveWithGuess (y, population);
  //  std::cout << "#iterations:     " << solver.iterations() << std::endl;
  //  std::cout << "estimated error: " << solver.error()      << std::endl;
  //}

  //assert (false);


  SparseLU <SparseMatrix<double>, COLAMDOrdering<int>> solver;
  //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;


  cout << "Analyzing system of rate equations..."      << endl;

  solver.analyzePattern (RT);

  cout << "Factorizing system of rate equations..."    << endl;

  solver.factorize (RT);

  if (solver.info() != Eigen::Success)
  {
    cout << "Factorization failed with error message:" << endl;
    cout << solver.lastErrorMessage()                  << endl;
    assert (false);
  }


  //cout << "Try compute" << endl;

  //solver.compute (RT);

  //if (solver.info() != Eigen::Success)
  //{
  //  cout << "Decomposition failed" << endl;
  //  //assert(false);
  //}


  cout << "Solving rate equations for the level populations..." << endl;

  population = solver.solve (y);

  if (solver.info() != Eigen::Success)
  {
    cout << "Solving failed with error:" << endl;
    cout << solver.lastErrorMessage()    << endl;
    assert (false);
  }

  cout << "Succesfully solved for the level populations!"       << endl;


  //OMP_PARALLEL_FOR (p, ncells)
  //{
  //
  //  for (long i = 0; i < linedata.nlev; i++)
  //  {
  //    const long I = index (p, i);

  //    population[I] = population_prev1[I];

  //    //if (population[I] < 1.0E-50)
  //    //{
  //    //  population[I] = 1.0E-50;
  //    //}
  //  }
  //}


}
