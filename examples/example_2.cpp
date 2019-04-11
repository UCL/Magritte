
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Io/io_Python.hpp"
#include "Simulation/simulation.hpp"
#include "Tools/logger.hpp"


int main (int argc, char **argv)
{

  if (argc != 2)
  {
    cout << "Please provide a model file as argument." << endl;
  }

  else
  {
    const string modelName = argv[1];

    cout << "Running model: " << modelName << endl;


    IoPython io ("hdf5", modelName);


    Simulation simulation;
    simulation.read (io);

    simulation.parameters.set_max_iter (10);
    simulation.parameters.set_pop_prec (1.0E-4);


    simulation.compute_spectral_discretisation ();

    simulation.compute_boundary_intensities ();

    simulation.compute_LTE_level_populations ();

    simulation.compute_level_populations ();


    simulation.write (io);

    //for (long p = 0; p < simulation.parameters.ncells(); p++)
    //{
    //  cout << simulation.lines.population[p][0](0) << "   " << simulation.lines.population[p][0](1) << endl;
    //}

    cout << "Done." << endl;
  }


  return (0);

}
