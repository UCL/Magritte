// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include <Eigen/Dense>

#include "catch.hpp"

#include "configure.hpp"
#include "Simulation/simulation.hpp"
#include "Io/cpp/io_cpp_text.hpp"


TEST_CASE ("LineProducingSpecies::update_using_statistical_equilibrium")
{

  // Setup


  const string model_folder = "/home/frederik/MagritteProjects/All/model_problem_2a/";
  //const string model_folder = string (MAGRITTE_FOLDER)
  //                            + "/data/testdata/model_test_lineProducingSpecies/";

  IoText io (model_folder);

  Simulation simulation;
  simulation.read (io);

  simulation.compute_LTE_level_populations ();



  // Set Jeff to 1 everywhere

  for (long p = 0; p < simulation.parameters.ncells(); p++)
  {
    for (int k = 0; k < simulation.lines.lineProducingSpecies[0].linedata.nrad; k++)
    {
      simulation.lines.lineProducingSpecies[0].Jeff[p][k] = 1.0;

      simulation.lines.lineProducingSpecies[0].lambda[p][k].nr.clear();
      simulation.lines.lineProducingSpecies[0].lambda[p][k].Ls.clear();
    }
  }


  for (int i = 0; i < simulation.lines.lineProducingSpecies[0].linedata.nlev; i++)
  {
    const long ind = simulation.lines.lineProducingSpecies[0].index (0, i);

    cout <<   simulation.lines.lineProducingSpecies[0].population[ind]
            / simulation.lines.lineProducingSpecies[0].population_tot[0] << endl;
  }

  // Run function

  simulation.lines.lineProducingSpecies[0].update_using_statistical_equilibrium (
    simulation.chemistry.species.abundance,
    simulation.thermodynamics.temperature.gas);





  SECTION ("Does the solution satisfy statistical equilibrium?")
  {
    LineProducingSpecies lspec = simulation.lines.lineProducingSpecies[0];

    cout << "nlev = " << lspec.linedata.nlev << endl;
    cout << "nrad = " << lspec.linedata.nrad << endl;

    // Get transition matrix

    const long p = 0;

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero (lspec.linedata.nlev, lspec.linedata.nlev);
    Eigen::VectorXd lhs (lspec.linedata.nlev);
    Eigen::VectorXd rhs (lspec.linedata.nlev);

    for (int k = 0; k < lspec.linedata.nrad; k++)
    {
      const long i = lspec.linedata.irad[k];
      const long j = lspec.linedata.jrad[k];

      R(i,j) = lspec.linedata.A[k] + lspec.linedata.Bs[k] * lspec.Jeff[p][k];
      R(j,i) =                       lspec.linedata.Ba[k] * lspec.Jeff[p][k];


      for (CollisionPartner &colpar : lspec.linedata.colpar)
      {
        double abn = simulation.chemistry.species.abundance[p][colpar.num_col_partner];
        double tmp = simulation.thermodynamics.temperature.gas[p];

        colpar.adjust_abundance_for_ortho_or_para (tmp, abn);
        colpar.interpolate_collision_coefficients (tmp);

        for (long k = 0; k < colpar.ncol; k++)
        {
          const long i = colpar.icol[k];
          const long j = colpar.jcol[k];

          colpar.Ce_intpld[k] = colpar.Cd_intpld[k]
                                * lspec.linedata.weight[i] / lspec.linedata.weight[j]
                                * exp ( - HH*lspec.linedata.frequency[k] / (KB*tmp) );

          R(i,j) += colpar.Cd_intpld[k] * abn;
          R(j,i) += colpar.Ce_intpld[k] * abn;
        }
      }
    }


    // Compute left hand side

    for (long i = 0; i < lspec.linedata.nlev; i++)
    {
      lhs(i) = 0.0;

      for (long j = 0; j < lspec.linedata.nlev; j++)
      {
        lhs(i) += lspec.population[lspec.index (p,j)] * R(j,i);
      }
    }


    // Compute right hand side

    for (long i = 0; i < lspec.linedata.nlev; i++)
    {
      rhs(i) = 0.0;

      for (long j = 0; j < lspec.linedata.nlev; j++)
      {
        rhs(i) += R (i,j);
      }

      rhs(i) *= lspec.population[lspec.index (p,i)];
    }


    // Check if left and right hand sides are equal

    cout << R << endl;

    for (long i = 0; i < lspec.linedata.nlev; i++)
    {
      const long ind = lspec.index (p, i);

      cout << "pop = " << lspec.population[ind] / lspec.population_tot[p] << "\t"
           << "lhs = " << lhs (i)               << "\t"
           << "rhs = " << rhs (i)               << endl;

      CHECK (lhs(i) == rhs(i));
    }

    double tot = 0.0;

    for (long i = 0; i < lspec.linedata.nlev; i++)
    {
      const long ind = lspec.index (p, i);

      tot += lspec.population[ind] / lspec.population_tot[p];
    }

    //CHECK (tot== 1.0);

  }

  CHECK (true);

  //SECTION ("Pass around populations")
  //{
  //  VectorXd pop (2);
  //  pop << 1.0, 2.0;

  //  for (long ind = 0; ind < lspec.population.size(); ind++)
  //  {
  //    CHECK (lspec.population_prev1[ind] == 0.0*pop[ind]);
  //    CHECK (lspec.population_prev2[ind] == 1.0*pop[ind]);
  //    CHECK (lspec.population_prev3[ind] == 2.0*pop[ind]);
  //  }
  //}


  //SECTION ("Populations solver")
  //{
  //  VectorXd pop (2);
  //  pop << 2.0/3.0, 1.0/3.0;

  //  for (long ind = 0; ind < lspec.population.size(); ind++)
  //  {
  //    CHECK (lspec.population[ind] == pop[ind]);
  //  }


  //}

}
