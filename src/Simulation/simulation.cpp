// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "simulation.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/logger.hpp"

#include "sim_spectrum.tpp"
#include "sim_radiation.tpp"
#include "sim_lines.tpp"


//#include "Raypair/rayblock.hpp"
//#include "Rayblock/rayBlock_v.hpp"
//#include "Raypair/rayblock.cuh"


//void Simulation :: compute_radiation_field ()
//{
//    /// Create a RayBlock
//    RayBlock *rayblock = new RayBlock (parameters.ncells(),
//                                       parameters.nfreqs(),
//                                       parameters.nlines(),
//                                       nraypairs,
//                                       geometry.max_npoints_on_rays);
//
//    /// Set model data
//    rayblock->copy_model_data (*this);
//
//
//    compute_radiation_field (rayblock);
//
//    /// Delete ray block
//    delete rayblock;
//}
//
//
//
//void Simulation :: compute_radiation_field (RayBlock *rayblock)
//{
//    // Initialisations
//    for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
//    {
//        lspec.lambda.clear ();
//    }
//
//    radiation.initialize_J ();
//
//    /// Set maximum number of points along a ray, if not set yet
//    if (geometry.max_npoints_on_rays == -1)
//    {
//        get_max_npoints_on_rays <CoMoving> ();
//    }
//
//
//
//
//    for (size_t rr = 0; rr < parameters.nrays()/2; rr++)
//    {
//        const size_t         RR = rr - MPI_start (parameters.nrays()/2);
//        const size_t         ar = geometry.rays.antipod[rr];
//        const double weight_ang = geometry.rays.weights[rr];
//
//        RayQueue rayqueue (rayblock->nraypairs);
//
//
//        logger.write ("ray = ", rr);
//
//        for (size_t o = 0; o < parameters.ncells(); o++)
//        {
//            const double dshift_max = get_dshift_max (o);
//
//            // Trace ray pair
//            const RayData ray_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
//            const RayData ray_rr = geometry.trace_ray <CoMoving> (o, rr, dshift_max);
//
//            const size_t depth = ray_ar.size() + ray_rr.size() + 1;
//
//            if (depth > 1)
//            {
//                /// Add ray pair to queue
//                rayqueue.add (ray_ar, ray_rr, o, depth);
//
//                if (rayqueue.complete)
//                {
//                    rayblock->setup (*this, RR, rr, rayqueue.get_complete_block());
//                    rayblock->solve ();
//                    rayblock->store (*this);
//                }
//            }
//            else
//            {
//                /// Extract radiation field from boundary
//                get_radiation_field_from_boundary (RR, rr, o);
//            }
//        }
//
//        /// Compute the unfinished rays in the queue
//        for (const ProtoRayBlock &prb : rayqueue.queue)
//        {
//            rayblock->nraypairs = prb.nraypairs();
//            rayblock->width     = prb.nraypairs() * parameters.nfreqs();
//
//            rayblock->setup (*this, RR, rr, prb);
//            rayblock->solve ();
//            rayblock->store (*this);
//        }
//    }
//
//}


int Simulation :: compute_radiation_field_cpu ()
{
    return cpu_compute_radiation_field (1, 1, 1, 1);
}



int Simulation :: cpu_compute_radiation_field (
        const size_t nraypairs,
        const size_t gpuBlockSize,
        const size_t gpuNumBlocks,
        const double inverse_dtau_max         )
{
    // Set timers
    Timer timer("compute radiation field (CPU)");
    timer.start();


    // Initialisations
    for (auto &lspec : lines.lineProducingSpecies) {lspec.lambda.clear();}

    radiation.initialize_J ();


    /// Set maximum number of points along a ray, if not set yet
    if (geometry.max_npoints_on_rays == -1)
    {
        get_max_npoints_on_rays <CoMoving> ();
    }

    cout << "maxnpoints on rays = " << geometry.max_npoints_on_rays << endl;


    // Get number of threads
    const size_t nthreads = get_nthreads();

    /// Create and initialize a solver fo each thread
    vector<cpuSolver*> solvers (nthreads);

    for (auto &solver : solvers)
    {
        // Create a sover object
        solver = new cpuSolver (parameters.ncells(), parameters.nfreqs(),
                                parameters.nlines(), parameters.nboundary(),
                                nraypairs,           geometry.max_npoints_on_rays,
                                parameters.n_off_diag);

        /// Set GPU block size
        solver->gpuBlockSize     = gpuBlockSize;
        solver->gpuNumBlocks     = gpuNumBlocks;
        solver->inverse_dtau_max = inverse_dtau_max;

        /// Set model data
        solver->copy_model_data (*this);
    }

    MPI_PARALLEL_FOR (rr, parameters.nrays()/2)
    {
        const size_t RR = rr - MPI_start (parameters.nrays()/2);
        const size_t ar = geometry.rays.antipod[rr];

        Queue queue (nraypairs);

//        cout << "complete = ";
//        if (rayqueue.complete()) {cout << "True"  << endl;}
//        else                     {cout << "False" << endl;}


        logger.write ("ray = ", rr);

#       pragma omp parallel default (shared)
        {
//            const size_t t = omp_get_thread_num();
            auto &solver = solvers[omp_get_thread_num()];


            for (size_t o = omp_get_thread_num(); o < parameters.ncells(); o += omp_get_num_threads())
            {
                const double dshift_max = get_dshift_max (o);

                // Trace ray pair
                const RayData ray_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
                const RayData ray_rr = geometry.trace_ray <CoMoving> (o, rr, dshift_max);

                const size_t depth = ray_ar.size() + ray_rr.size() + 1;

                if (depth > 1)
                {
                    bool       completed;
                    ProtoBlock complete_block;

#                   pragma omp critical (update_queue)
                    {
                        queue.add (ray_ar, ray_rr, o, depth);
                        completed = queue.some_are_completed();

                        if (completed) complete_block = queue.get_complete_block();
                    }

                    if (completed)
                    {
                        solver->solve (complete_block, RR, rr, *this);

#                       pragma omp critical (update_Lambda)
                        {
                            solver->update_Lambda (*this);
                        }
                    }

                }
                else
                {
                    /// Extract radiation field from boundary
                    get_radiation_field_from_boundary (RR, rr, o);
                }
            }
        }

        /// Compute the unfinished rays in the queue
//        for (long s = omp_get_thread_num(); s < rayqueue.queue.size(); s += omp_get_num_threads())
        for (const ProtoBlock &prb : queue.queue)
        {
//            const ProtoRayBlock &prb = rayqueue.queue[s];
//
            solvers[0]->nraypairs = prb.nraypairs();
            solvers[0]->width     = prb.nraypairs() * parameters.nfreqs();

            solvers[0]->solve (prb, RR, rr, *this);
        }
    }


    // Gather and reduce results of all MPI processes to get Lambda and J
#   if (MPI_PARALLEL)
        logger.write ("Gathering Lambda operators...");
        for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
        {
//            lspec.lambda.MPI_gather ();
        }
        logger.write ("Reducing the mean intensities (J's)...");
//        radiation.MPI_reduce_J ();
#   endif


    if (parameters.use_scattering())
    {
//        radiation.calc_U_and_V();
    }


//    Ld.clear();
//    for (int n = 0; n < parameters.ncells(); n++)
//    {
//        Ld.push_back(solvers[0]->L_diag[n]);
//    }
//
//    Lu.resize(solvers[0]->n_off_diag);
//    Ll.resize(solvers[0]->n_off_diag);
//
//    for (int m = 0; (m < solvers[0]->n_off_diag; m++)
//    {
//        Lu[m].clear();
//        Ll[m].clear();
//
//        for (int n = 0; n < parameters.ncells(); n++)
//        {
//            Lu[m].push_back(solvers[0]->L_upper[solvers[0]->M(m,n)]);
//            Ll[m].push_back(solvers[0]->L_lower[solvers[0]->M(m,n)]);
//        }
//    }


    /// Delete solvers
    for (auto &solver : solvers) {delete solver;}

    // Stop timer and print results
    timer.stop();
    timer.print();

    return (0);
}
