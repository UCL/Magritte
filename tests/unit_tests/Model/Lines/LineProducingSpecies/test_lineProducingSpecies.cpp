// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "configure.hpp"
#include "Model/parameters.hpp"
#include "Model/Lines/LineProducingSpecies/lineProducingSpecies.hpp"
#include "Io/io_Python.hpp"


TEST_CASE ("LineProducingSpecies::update_using_statistical_equilibrium")
{

  // Setup

  const string model_folder = string (MAGRITTE_FOLDER)
                              + "/data/testdata/model_test_lineProducingSpecies.hdf5";

  IoPython io ("hdf5", model_folder);

  Parameters parameters;
  parameters.read (io);

  LineProducingSpecies lineProducingSpecies;
  lineProducingSpecies.read (io, 0, parameters);

  Double1 temperature {10.0};
  Double2 abundance   {{1000.0, 1000.0}};


  // Run function

  lineProducingSpecies.update_using_statistical_equilibrium (
    abundance,
    temperature                                             );


  SECTION ("Pass around populations")
  {
    VectorXd pop (2);
    pop << 1.0, 2.0;

    for (long ind = 0; ind < lineProducingSpecies.population.size(); ind++)
    {
      CHECK (lineProducingSpecies.population_prev1[ind] == 0.0*pop[ind]);
      CHECK (lineProducingSpecies.population_prev2[ind] == 1.0*pop[ind]);
      CHECK (lineProducingSpecies.population_prev3[ind] == 2.0*pop[ind]);
    }
  }


  SECTION ("Populations solver")
  {
    VectorXd pop (2);
    pop << 2.0/3.0, 1.0/3.0;

    for (long ind = 0; ind < lineProducingSpecies.population.size(); ind++)
    {
      CHECK (lineProducingSpecies.population[ind] == pop[ind]);
    }


  }

}
