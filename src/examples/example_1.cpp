// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "configure.hpp"
#include "Io/python/io_python.hpp"
#include "Simulation/simulation.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/logger.hpp"


int main (int argc, char **argv)
{
    // Create a logger and a timer
    Logger logger ("example_1");
    Timer  timer  ("overall_1");

    // Check input arguments
    if (argc != 2)
    {
      logger.write("Please provide a model file as argument.");
      return (-1);
    }

    // Start timer
    timer.start();

    // Store model name
    const string modelName = argv[1];

    logger.write("Running model: " + modelName);

#   if (MPI_PARALLEL)
        MPI_Init (NULL, NULL);
#   endif

    IoPython io ("hdf5", modelName);

    Simulation simulation;
    simulation.read (io);

    simulation.compute_spectral_discretisation ();
    simulation.compute_boundary_intensities    ();
    simulation.compute_LTE_level_populations   ();
    simulation.compute_level_populations       (io);

    simulation.write (io);

#   if (MPI_PARALLEL)
        MPI_Finalize ();
#   endif

    timer.stop();
    logger.write(timer.get_print_total_string());

    return (0);
}
