
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
    Logger logger ("example_6_gpu_benchmark");

    /// Error if no model file was providad as argument
    if (argc != 4)
    {
      logger.write ("Please provide a model file, a number of ray pairs, and a GPU block size as argument."); return (-1);
    }

    /// Store model name
    const string modelName  =           argv[1];
    const long nraypairs    = std::atol(argv[2]);
    const long gpuBlockSize = std::atol(argv[3]);

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
    simulation.gpu_get_device_properties();

    /// Read model data
    simulation.read (io);

    simulation.compute_spectral_discretisation ();
    simulation.compute_boundary_intensities    ();
    simulation.compute_LTE_level_populations   ();


    timer0.start();
    /// Set maximum number of points along a ray, if not set yet
    if (simulation.geometry.max_npoints_on_rays == -1)
    {
        simulation.get_max_npoints_on_rays <CoMoving> ();
    }
    timer0.stop();


    /// Create a gpuRayPair object
    RayBlock **rayblocks = new RayBlock*[get_nthreads()];

    for (int i = 0; i < get_nthreads(); i++)
    {
        rayblocks[i] = new RayBlock(simulation.parameters.ncells(),
                                    simulation.parameters.nfreqs(),
                                    simulation.parameters.nlines(),
                                    nraypairs,
                                    simulation.geometry.max_npoints_on_rays);

        /// Set GPU block size
        rayblocks[i]->gpuBlockSize = gpuBlockSize;

        /// Set model data
        rayblocks[i]->copy_model_data(simulation);
    }





    for (size_t rr = 0; rr < simulation.parameters.nrays() / 2; rr++)
    {
        const size_t RR = rr - MPI_start(simulation.parameters.nrays() / 2);
        const size_t ar = simulation.geometry.rays.antipod[rr];
        const double weight_ang = simulation.geometry.rays.weights[rr];

        RayQueue rayqueue(nraypairs);

        timer5.start();


#       pragma omp parallel default (shared)
        {
            RayBlock *rayblock = rayblocks[omp_get_thread_num()];

            logger.write("ray = ", rr);


//            for (size_t o = 0; o < simulation.parameters.ncells(); o++)
            for (long o = omp_get_thread_num(); o < simulation.parameters.ncells(); o += omp_get_num_threads())
            {
                timer1.start();
                const double dshift_max = simulation.get_dshift_max(o);

                // Trace ray pair
                const RayData raydata_ar = simulation.geometry.trace_ray<CoMoving>(o, ar, dshift_max);
                const RayData raydata_rr = simulation.geometry.trace_ray<CoMoving>(o, rr, dshift_max);

                const size_t depth = raydata_ar.size() + raydata_rr.size() + 1;
                timer1.stop();

                if (depth > 1)
                {
                    /// Add ray pair to queue
                    rayqueue.add(raydata_ar, raydata_rr, o, depth);

                    if (rayqueue.complete)
                    {
                        timer2.start();
                        rayblock->setup(simulation, RR, rr, rayqueue.get_complete_block());
                        timer2.stop();

                        timer3.start();
                        rayblock->solve();
                        timer3.stop();
                        timer3.print();

                        timer4.start();
                        rayblock->store(simulation);
                        timer4.stop();
                    }
                }
                else
                {
                    /// Extract radiation field from boundary
                    simulation.get_radiation_field_from_boundary(RR, rr, o);
                }
            }
            timer5.stop();

            /// Compute the unfinished rays in the queue
            timer9.start();
            for (const ProtoRayBlock &prb : rayqueue.queue)
            {
                rayblock->nraypairs = prb.nraypairs();
                rayblock->width = prb.nraypairs() * simulation.parameters.nfreqs();

                timer6.start();
                rayblock->setup(simulation, RR, rr, prb);
                timer6.stop();

                timer7.start();
                rayblock->solve();
                timer7.stop();
                timer7.print();

                timer8.start();
                rayblock->store(simulation);
                timer8.stop();
            }
            timer9.stop();
        }
    }

//    rayblock->timer0.print_total();
//    rayblock->timer1.print_total();
//    rayblock->timer2.print_total();
//    rayblock->timer3.print_total();



    /// Delete ray block


    for (int i = 0; i < get_nthreads(); i++)
    {
        delete rayblocks[i];
    }
    delete rayblocks;


    /// Write output
//    simulation.write (io);

    /// Print final timers
    timer0.print_total();
    timer1.print_total();
    timer2.print_total();
    timer3.print_total();
    timer4.print_total();
    timer5.print_total();
    timer6.print_total();
    timer7.print_total();
    timer8.print_total();
    timer9.print_total();

    /// Write exit message
    logger.write ("--- Magritte example 6 GPU benchmark is done.");

    return (0);
}
