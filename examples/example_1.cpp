
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


    IoPython io ("hdf5", modelName);


    Simulation simulation;
    simulation.read (io);


    simulation.compute_spectral_discretisation ();

    simulation.compute_boundary_intensities ();

    simulation.compute_LTE_level_populations ();

    simulation.compute_radiation_field ();


    simulation.write (io);

    for (long i = 0; i < simulation.lines.emissivity.size(); i++)
    {
      //std::cout << simulation.lines.emissivity[i] << std::endl;
      //std::cout << simulation.lines.opacity[i] << std::endl;
    }

    std::cout << "Done." << std::endl;
  }


  return (0);

}
