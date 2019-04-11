
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


    simulation.compute_spectral_discretisation ();

    simulation.compute_boundary_intensities ();

    simulation.compute_LTE_level_populations ();

    simulation.compute_radiation_field ();


    simulation.write (io);

    for (long i = 0; i < simulation.lines.emissivity.size(); i++)
    {
      //cout << simulation.lines.emissivity[i] << endl;
      //cout << simulation.lines.opacity[i]    << endl;
    }

    cout << "Done." << endl;
  }


  return (0);

}
