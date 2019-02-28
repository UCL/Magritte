
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
using std::string;

#include "Io/io_Python.hpp"
#include "Simulation/simulation.hpp"


int main (int argc, char **argv)
{

  if (argc != 2)
  {
    std::cout << "Please provide a model file as argument." << std::endl;
  }

  else
  {
    const string modelName = argv[1];

    std::cout << "Running model: " << modelName << std::endl;


    IoPython io ("io_hdf5", modelName);


    Simulation simulation;
    simulation.read (io);

    simulation.parameters.set_max_iter (100);
    simulation.parameters.set_pop_prec (1.0E-4);


    simulation.compute_spectral_discretisation ();

    simulation.compute_boundary_intensities ();

    simulation.compute_level_populations ();


    simulation.write (io);

    //for (long p = 0; p < simulation.parameters.ncells(); p++)
    //{
    //  std::cout << simulation.lines.population[p][0](0) << "   " << simulation.lines.population[p][0](1) << std::endl;
    //}

    std::cout << "Done." << std::endl;
  }


  return (0);

}
