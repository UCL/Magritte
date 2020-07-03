// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "configure.hpp"
#include "Io/python/io_python.hpp"
#include "Simulation/simulation.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/logger.hpp"


int main (int argc, char **argv)
{
    // Create a logger
    Logger logger ("benchmark_1_radiation");
    
    // Create timers
    Timer  timer_total ("total");
    Timer  timer_calcs ("calcs");
    

    // Check input arguments
    if (argc != 2)
    {
      logger.write("Please provide a model file as argument.");
      return (-1);
    }

    // Store model name
    const string modelName = argv[1];
    
#   if (MPI_PARALLEL)
    MPI_Init (NULL, NULL);
#   endif

    logger.write ("----------------------------------");
    logger.write (" Magritte Benchmark 1 (radiation) ");
    logger.write ("----------------------------------");
    logger.write (" Running model: " + modelName      );
    logger.write ("----------------------------------");
    logger.write ("MPI comm size = ", MPI_comm_size ());
    logger.write ("MPI comm rank = ", MPI_comm_rank ());
    logger.write ("OMP # threads = ",  get_nthreads ());
    logger.write ("----------------------------------");

    timer_total.start ();

    IoPython io ("hdf5", modelName);
    
    Simulation simulation;
    simulation.read (io);
    
    timer_calcs.start ();

    simulation.compute_spectral_discretisation ();
    simulation.compute_boundary_intensities    ();
    simulation.compute_LTE_level_populations   ();
    simulation.compute_radiation_field_gpu     ();

    timer_calcs.stop ();

    simulation.write (io);
    
    timer_total.stop ();

#   if (MPI_PARALLEL)
        MPI_Finalize ();
#   endif
    
    logger.write(timer_calcs.get_print_total_string());
    logger.write(timer_total.get_print_total_string());

    return (0);
}
