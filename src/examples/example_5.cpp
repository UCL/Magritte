
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "configure.hpp"
#include "Io/cpp/io_cpp_text.hpp"
//#include "Io/python/io_python.hpp"
#include "Simulation/simulation.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/logger.hpp"


int main (int argc, char **argv)
{

  if (argc != 2)
  {
    cout << "(Magritte): Please provide a model file as argument." << endl;
  }

  else
  {
    const string modelName = argv[1];

    cout << "---------------------------------------------------" << endl;
    cout << "                                                   " << endl;
    cout << " ___Magritte______________________________________ " << endl;
    cout << "|                                                 |" << endl;
    cout << "| Imager tests                                    |" << endl;
    cout << "|_________________________________________________|" << endl;
    cout << "                                                   " << endl;
    cout << "---------------------------------------------------" << endl;
    cout << "  Running model: " << modelName                      << endl;
    cout << "---------------------------------------------------" << endl;


#   pragma omp parallel
    {
      if (omp_get_thread_num() == 0)
      {
        cout << "---------------------------------------------------" << endl;
        cout << "  n_omp_threads = " << omp_get_num_threads ()        << endl;
        cout << "  n_simd_lanes  = " << n_simd_lanes                  << endl;
        cout << "---------------------------------------------------" << endl;
      }
    }


    const double width = 150.0 / CC;


    //IoPython io ("hdf5", modelName);
    IoText io (modelName);

    Simulation simulation;

    simulation.parameters.set_use_scattering (false);

    simulation.read (io);

    simulation.compute_spectral_discretisation_image (width);
    simulation.compute_boundary_intensities ();
    simulation.compute_LTE_level_populations ();

    simulation.compute_and_write_image (io, 0);

    cout << "---------------------------------------------------" << endl;
    cout << "  Done.                                            " << endl;
    cout << "---------------------------------------------------" << endl;
  }


  return (0);

}
