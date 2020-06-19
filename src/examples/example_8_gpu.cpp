
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "configure.hpp"
#include "Io/cpp/io_cpp_text.hpp"
#include "Io/python/io_python.hpp"
#include "Simulation/simulation.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/logger.hpp"
#include "Tools/timer.hpp"


/// Test bench for performance testing of the setup and solve functions
/// that compute the radiation field in Magritte.
///////////////////////////////////////////////////////////////////////

int main (int argc, char **argv)
{
    /// Create a logger
    Logger logger ("example_8_gpu_benchmark");

    /// Error if no model file was providad as argument
    if (argc != 5)
    {
      logger.write ("Please provide a model file, a number of ray pairs, and a GPU block size as argument."); return (-1);
    }

    /// Store model name
    const string modelName  =           argv[1];
    const long nraypairs    = std::atol(argv[2]);
    const long gpuBlockSize = std::atol(argv[3]);
    const long gpuNumBlocks = std::atol(argv[4]);
    const double inverse_dtau_max = 1.0;
    const size_t n_off_diag = 10000;

    logger.write_line (                                                   );
    logger.write      ("   Magritte   (GPU)"                              );
    logger.write_line (                                                   );
    logger.write      ("Performance tests for setup and solver functions.");
    logger.write      ("( Only for GPU and single node use, i.e. no MPI )");
    logger.write_line (                                                   );
    logger.write      ("Running model: " + modelName                      );
    logger.write_line (                                                   );
    logger.write      ("nraypairs = ",     nraypairs                      );
    logger.write_line (                                                   );

    /// Create timer instances
    Timer timer0 ("prepa");
    Timer timer1 ("trace");
    Timer timer2 ("setup");
    Timer timer3 ("solve");
    Timer timer4 ("store");
    Timer timer5 ("total");
    Timer timer6 ("pp_setup");
    Timer timer7 ("pp_solve");
    Timer timer8 ("pp_store");
    Timer timer9 ("gg_total");

    IoPython io ("hdf5", modelName);
//    IoText io (modelName);

    Simulation simulation;

    /// Write gpu properties
//    simulation.gpu_get_device_properties();

    /// Read model data
    simulation.read (io);

    simulation.compute_spectral_discretisation ();
    simulation.compute_boundary_intensities    ();


    for (long n = 10; n < 1.0e+7; n=10*n)
    {
        simulation.parameters.n_off_diag = n;
        simulation.compute_LTE_level_populations   ();

        simulation.compute_radiation_field_cpu          ();
        simulation.compute_Jeff                         ();
        simulation.compute_level_populations_from_stateq();
    }



//    // Initialisations
//    for (auto &lspec : simulation.lines.lineProducingSpecies)
//    {
//        lspec.lambda.clear ();
//    }
//
//    simulation.radiation.initialize_J ();
//
//    /// Set maximum number of points along a ray, if not set yet
//    if (simulation.geometry.max_npoints_on_rays == -1)
//    {
//        simulation.get_max_npoints_on_rays <CoMoving> ();
//    }
//
//    // Get number of threads
//    const size_t nthreads = get_nthreads();
//    cout << "nthreads = " << nthreads << endl;
//
//    /// Create and initialize a solver fo each thread
//    vector<cpuSolver*> solvers (nthreads);
//
//    for (auto &solver : solvers)
//    {
//        // Create a sover object
//        solver = new cpuSolver (simulation.parameters.ncells(),
//                                simulation.parameters.nfreqs(),
//                                simulation.parameters.nlines(),
//                                nraypairs,
//                                simulation.geometry.max_npoints_on_rays,
//                                n_off_diag);
//
//        /// Set GPU block size
//        solver->gpuBlockSize     = gpuBlockSize;
//        solver->gpuNumBlocks     = gpuNumBlocks;
//        solver->inverse_dtau_max = inverse_dtau_max;
//
//        /// Set model data
//        solver->copy_model_data (simulation);
//    }
//
//
//    for (size_t rr = 0; rr < simulation.parameters.nrays()/2; rr++)
//    {
//        const size_t RR = rr - MPI_start (simulation.parameters.nrays()/2);
//        const size_t ar = simulation.geometry.rays.antipod[rr];
//
//        Queue queue (nraypairs);
//
////        cout << "complete = ";
////        if (rayqueue.complete()) {cout << "True"  << endl;}
////        else                     {cout << "False" << endl;}
//
//
//        logger.write ("ray = ", rr);
//
////#       pragma omp parallel default (shared)
//        {
////            const size_t t = omp_get_thread_num();
//            auto &solver = solvers[omp_get_thread_num()];
//
//
//            for (size_t o = omp_get_thread_num(); o < simulation.parameters.ncells(); o += omp_get_num_threads())
//            {
//                const double dshift_max = simulation.get_dshift_max (o);
//
//                // Trace ray pair
//                const RayData ray_ar = simulation.geometry.trace_ray <CoMoving> (o, ar, dshift_max);
//                const RayData ray_rr = simulation.geometry.trace_ray <CoMoving> (o, rr, dshift_max);
//
//                const size_t depth = ray_ar.size() + ray_rr.size() + 1;
//
//                if (depth > 1)
//                {
//                    bool       completed;
//                    ProtoBlock complete_block;
//
//#                   pragma omp critical
//                    {
//                        queue.add (ray_ar, ray_rr, o, depth);
//
//                        completed = queue.some_are_completed();
//
//                        if (completed) complete_block = queue.get_complete_block();
//                    }
//
//                    if (completed)
//                    {
//                        solver->solve (complete_block, RR, rr, simulation);
//                    }
//                }
//                else
//                {
//                    /// Extract radiation field from boundary
//                    simulation.get_radiation_field_from_boundary (RR, rr, o);
//                }
//            }
//        }
//
//        /// Compute the unfinished rays in the queue
////        for (long s = omp_get_thread_num(); s < rayqueue.queue.size(); s += omp_get_num_threads())
//        for (const ProtoBlock &prb : queue.queue)
//        {
////            const ProtoRayBlock &prb = rayqueue.queue[s];
//
//            solvers[0]->nraypairs = prb.nraypairs();
//            solvers[0]->width     = prb.nraypairs() * simulation.parameters.nfreqs();
//
//            solvers[0]->solve (prb, RR, rr, simulation);
//        }
//    }
//
//
//    /// Delete solvers
//    for (auto &solver : solvers) {delete solver;}
//
//
//    /// Write output
////    simulation.write (io);
//
//    /// Print final timers
//    timer0.print_total();
//    timer1.print_total();
//    timer2.print_total();
//    timer3.print_total();
//    timer4.print_total();
//    timer5.print_total();
//    timer6.print_total();
//    timer7.print_total();
//    timer8.print_total();
//    timer9.print_total();
//
//    /// Write exit message
//    logger.write ("--- Magritte example 8 GPU benchmark is done.");

    return (0);
}
