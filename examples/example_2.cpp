
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
    cout << "Please provide a model file as argument." << endl;
  }

  else
  {
    const string modelName = argv[1];

    cout << "Running model: " << modelName << endl;

    const bool use_Ng_acceleration = true;
    const long max_niterations     = 20;


#   pragma omp parallel
    {
      cout << "n_omp_threads = " << omp_get_num_threads () << endl;
    }


    cout << "n_simd_lanes = " << n_simd_lanes << endl;


#   if (MPI_PARALLEL)

      MPI_Init (NULL, NULL);

#   endif


    //IoPython io ("hdf5", modelName);
    IoText io (modelName);

    Simulation simulation;

    simulation.parameters.set_pop_prec       (1.0E-6);
    simulation.parameters.set_use_scattering (false);

    simulation.read (io);

    simulation.compute_spectral_discretisation ();

    simulation.compute_boundary_intensities ();

    simulation.compute_LTE_level_populations ();

    simulation.compute_level_populations_opts (io, use_Ng_acceleration, max_niterations);

    simulation.write (io);


#   if (MPI_PARALLEL)

      MPI_Finalize ();

#   endif


    cout << "Done." << endl;
  }


  return (0);

}
