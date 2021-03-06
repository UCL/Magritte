#include "Simulation/simulation.hpp"
#include "Simulation/Solver/gpu/gpu_solver.cuh"
//#include "Raypair/raypair.cuh"
//#include "Raypair/rayblock.cuh"


int Simulation :: handleCudaError (cudaError_t error)
{
    if (error != cudaSuccess)
    {
        logger.write ("CUDA ERROR : " + string (cudaGetErrorString (error)));
    }

    return (0);
}


int Simulation :: gpu_get_device_properties (void)
{
    int nDevices;
    HANDLE_ERROR (cudaGetDeviceCount (&nDevices));

    logger.write_line (                                      );
    logger.write      (" Properties of the available GPU's :");
    logger.write_line (                                      );

    for (long i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        HANDLE_ERROR (cudaGetDeviceProperties (&prop, i));

        const string cr  = std::to_string(prop.memoryClockRate);
        const string bw  = std::to_string(prop.memoryBusWidth);
        const string cc  = std::to_string(prop.major)+"."+std::to_string(prop.minor);
        const double pmb = 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;

        logger.write_line (                                                     );
        logger.write      ("Device Number                : ",  i                );
        logger.write      ("Device name                  : " + string(prop.name));
        logger.write      ("Compute compatibility        : " + cc               );
        logger.write      ("Memory Clock Rate (KHz)      : " + cr               );
        logger.write      ("Memory Bus Width (bits)      : " + bw               );
        logger.write      ("Peak Memory Bandwidth (GB/s) : ",  pmb              );
        logger.write_line (                                                     );
    }

    return (0);
}


//int Simulation :: gpu_compute_radiation_field (void)
//{
//  // Initialisations
//  for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
//  {
//    lspec.lambda.clear ();
//  }
//
//  radiation.initialize_J ();
//
//  /// Set maximum number of points along a ray, if not set yet
//  if (geometry.max_npoints_on_rays == -1)
//  {
//    get_max_npoints_on_rays <CoMoving> ();
//  }
//
//  /// Create a gpuRayPair object
//  gpuRayPair *raypair = new gpuRayPair (geometry.max_npoints_on_rays,
//                                        parameters.ncells(),
//                                        parameters.nfreqs(),
//                                        parameters.nlines()          );
//
//  /// Set model data
//  raypair->copy_model_data (Simulation(*this));
//
//
//  for (long r = 0; r < parameters.nrays()/2; r++)
//  {
//    const long R = r - MPI_start (parameters.nrays()/2);
//
//    logger.write ("ray = ", r);
//
//    for (long o = 0; o < parameters.ncells(); o++)
//    {
//      const long           ar = geometry.rays.antipod[r];
//      const double weight_ang = geometry.rays.weights[r];
//      const double dshift_max = get_dshift_max (o);
//
//
//      // Trace ray pair
//      const RayData raydata_r  = geometry.trace_ray <CoMoving> (o, r,  dshift_max);
//      const RayData raydata_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
//
//      if (raydata_r.size() + raydata_ar.size() > 0)
//      {
//        /// Setup such that the first ray is the longest (for performance)
//        raypair->setup (*this, raydata_ar, raydata_r, R, o);
//        /// Solve radiative transfer along ray pair
//        raypair->solve ();
//        /// Extract model data
//        raypair->extract_radiation_field (*this, R, r, o);
//      }
//      else
//      {
//        /// Extract radiation field from boundary consitions
//        get_radiation_field_from_boundary (R, r, o);
//      }
//    }
//  }
//
//  /// Delete raypair
//  delete raypair;
//
//  return (0);
//}


//int Simulation :: cpu_compute_radiation_field (const double inverse_dtau_max)
//{
//    // Initialisations
//    for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
//    {
//        lspec.lambda.clear ();
//    }

//    radiation.initialize_J ();

//    /// Set maximum number of points along a ray, if not set yet
//    if (geometry.max_npoints_on_rays == -1)
//    {
//        get_max_npoints_on_rays <CoMoving> ();
//    }

//    const long nraypairs = 1;

//    /// Create a gpuRayPair object
//    RayBlock_v *rayblock = new RayBlock_v (parameters.ncells(),
//                                           parameters.nfreqs(),
//                                           parameters.nlines(),
//                                           nraypairs,
//                                           geometry.max_npoints_on_rays);

//    /// Set inverse maximum optical depth increment
//    rayblock->inverse_dtau_max = inverse_dtau_max;

//    /// Set model data
//    rayblock->copy_model_data (*this);


//    for (size_t rr = 0; rr < parameters.nrays()/2; rr++)
//    {
//        const size_t RR = rr - MPI_start (parameters.nrays()/2);
//        const size_t ar = geometry.rays.antipod[rr];

//        logger.write ("ray = ", rr);

//        for (size_t o = 0; o < parameters.ncells(); o++)
//        {
//            const double dshift_max = get_dshift_max (o);

//            // Trace ray pair
//            const RayData ray_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
//            const RayData ray_rr = geometry.trace_ray <CoMoving> (o, rr, dshift_max);

//            const size_t depth = ray_ar.size() + ray_rr.size() + 1;

//            if (depth > 1)
//            {
//                const ProtoRayBlock_v prb (ray_ar, ray_rr, o);

//                rayblock->setup (*this, RR, rr, ProtoRayBlock_v (ray_ar, ray_rr, o));
//                rayblock->solve ();
//                rayblock->store (*this);
//            }
//            else
//            {
//                /// Extract radiation field from boundary
//                get_radiation_field_from_boundary (RR, rr, o);
//            }
//        }
//    }

//    /// Delete ray block
//    delete rayblock;

//    return (0);
//}




//int Simulation :: gpu_compute_radiation_field_2 (
//        const size_t nraypairs,
//        const size_t gpuBlockSize,
//        const size_t gpuNumBlocks,
//        const double inverse_dtau_max             )
//{
//    // Set timers
//    Timer timer("GPU compute radiation field");
//    timer.start();
//
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
//    // Get number of threads
//    const size_t nthreads = get_nthreads();
//
//    vector<RayBlock*> rayblocks (nthreads);
//
//    for (auto &rayblock : rayblocks)
//    {
//        // Create a RayBlock object
//        rayblock = new RayBlock (parameters.ncells(),
//                                 parameters.nfreqs(),
//                                 parameters.nlines(),
//                                 nraypairs,
//                                 geometry.max_npoints_on_rays);
//
//        /// Set GPU block size
//        rayblock->gpuBlockSize     = gpuBlockSize;
//        rayblock->gpuNumBlocks     = gpuNumBlocks;
//        rayblock->inverse_dtau_max = inverse_dtau_max;
//
//        /// Set model data
//        rayblock->copy_model_data (*this);
//    }
//
//
//    for (size_t rr = 0; rr < parameters.nrays()/2; rr++)
//    {
//        const size_t RR = rr - MPI_start (parameters.nrays()/2);
//        const size_t ar = geometry.rays.antipod[rr];
//
//        RayQueue rayqueue (nraypairs);
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
//            auto &rayblock = rayblocks[omp_get_thread_num()];
//
//
//            for (size_t o = omp_get_thread_num(); o < parameters.ncells(); o += omp_get_num_threads())
//            {
//                const double dshift_max = get_dshift_max (o);
//
//                // Trace ray pair
//                const RayData ray_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
//                const RayData ray_rr = geometry.trace_ray <CoMoving> (o, rr, dshift_max);
//
//                const size_t depth = ray_ar.size() + ray_rr.size() + 1;
//
//                if (depth > 1)
//                {
//#                   pragma omp critical (add_to_queue)
//                    {
//                        /// Add ray pair to queue
//                        rayqueue.add (ray_ar, ray_rr, o, depth);
//                    }
//
//#                   pragma omp critical (offload_to_gpu)
//                    {
//                        if (rayqueue.some_are_completed())
//                        {
//                            rayblock->solve_cpu (rayqueue.get_complete_block(), RR, rr, *this);
//                        }
//                    }
//                }
//                else
//                {
//                    /// Extract radiation field from boundary
//                    get_radiation_field_from_boundary (RR, rr, o);
//                }
//            }
//        }
//
//        /// Compute the unfinished rays in the queue
////        for (long s = omp_get_thread_num(); s < rayqueue.queue.size(); s += omp_get_num_threads())
//        for (const ProtoRayBlock &prb : rayqueue.queue)
//        {
////            const ProtoRayBlock &prb = rayqueue.queue[s];
//
//            rayblocks[0]->nraypairs = prb.nraypairs();
//            rayblocks[0]->width     = prb.nraypairs() * parameters.nfreqs();
//
//            rayblocks[0]->solve_cpu (prb, RR, rr, *this);
//        }
//    }
//
//
//    /// Delete ray blocks
//    for (auto &rayblock : rayblocks) {delete rayblock;}
//
//    // Stop timer and print results
//    timer.stop();
//    timer.print();
//
//    return (0);
//}


int Simulation :: gpu_compute_radiation_field_2 (
        const size_t nraypairs,
        const size_t gpuBlockSize,
        const size_t gpuNumBlocks,
        const double inverse_dtau_max           )
{
    // Set timers
    Timer timer("GPU compute radiation field");
    timer.start();


    // Initialisations
    for (auto &lspec : lines.lineProducingSpecies) {lspec.lambda.clear ();}

    radiation.initialize_J ();


    /// Set maximum number of points along a ray, if not set yet
    if (geometry.max_npoints_on_rays == -1)
    {
        get_max_npoints_on_rays <CoMoving> ();
    }

    // Get number of threads
    const size_t nthreads = get_nthreads();

    size_t ngpu = 0;
    size_t ncpu;
    if (nthreads > ngpu) ncpu = nthreads - ngpu;
    else                 ncpu = 0;

    /// Create and initialize a solver for each thread
    vector<gpuSolver*> gpuSolvers (ngpu);
    vector<cpuSolver*> cpuSolvers (ncpu);

    for (auto &solver : gpuSolvers)
    {
        // Create a sover object
        solver = new gpuSolver (parameters.ncells(), parameters.nfreqs(),
                                parameters.nlines(), parameters.nboundary(),
                                nraypairs,           geometry.max_npoints_on_rays,
                                parameters.n_off_diag);

        /// Set GPU block size
        solver->gpuBlockSize     = gpuBlockSize;
        solver->gpuNumBlocks     = gpuNumBlocks;

        /// Set model data
        solver->copy_model_data (*this);
    }

    for (auto &solver : cpuSolvers)
    {
        // Create a sover object
        solver = new cpuSolver (parameters.ncells(), parameters.nfreqs(),
                                parameters.nlines(), parameters.nboundary(),
                                nraypairs,           geometry.max_npoints_on_rays,
                                parameters.n_off_diag);

        /// Set GPU block size
        solver->gpuBlockSize     = gpuBlockSize;
        solver->gpuNumBlocks     = gpuNumBlocks;

        /// Set model data
        solver->copy_model_data (*this);
    }


    for (size_t rr = 0; rr < parameters.nrays()/2; rr++)
    {
        logger.write ("ray = ", rr);

        const size_t RR = rr - MPI_start (parameters.nrays()/2); // index ray
        const size_t ar = geometry.rays.antipod[rr];             // index antipod

        Queue queue (nraypairs);

//        cout << "complete = ";
//        if (rayqueue.complete()) {cout << "True"  << endl;}
//        else                     {cout << "False" << endl;}

#       pragma omp parallel default (shared)
        {
            // gpu solvers first, then cpu solvers
            const size_t t = omp_get_thread_num();

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

                    #pragma omp critical (update_queue)
                    {
                        queue.add (ray_ar, ray_rr, o, depth);
                        completed = queue.some_are_completed();

                        if (completed) complete_block = queue.get_complete_block();
                    }

                    if (completed)
                    {
                        if (t < ngpu)
                        {
                            gpuSolver *solver = gpuSolvers[t];

                            solver->solve (complete_block, RR, rr, *this);

                            #pragma omp critical (update_Lambda)
                            {
                                solver->update_Lambda (*this);
                            }
                        }
                        else
                        {
                            cpuSolver *solver = cpuSolvers[t-ngpu];

                            solver->solve (complete_block, RR, rr, *this);

                            #pragma omp critical (update_Lambda)
                            {
                                solver->update_Lambda (*this);
                            }
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

            cpuSolvers[0]->nraypairs = prb.nraypairs();
            cpuSolvers[0]->width     = prb.nraypairs() * parameters.nfreqs();

            cpuSolvers[0]->solve (prb, RR, rr, *this);
        }
    }


    // Gather and reduce results of all MPI processes to get Lambda and J
#   if (MPI_PARALLEL)
        logger.write ("Gathering Lambda operators...");
        for (auto &lspec : lines.lineProducingSpecies) {lspec.lambda.MPI_gather ();}
        logger.write ("Reducing the mean intensities (J's)...");
        radiation.MPI_reduce_J ();
#   endif


    /// Delete solvers
    for (auto &solver : gpuSolvers) {delete solver;}
    for (auto &solver : cpuSolvers) {delete solver;}

    // Stop timer and print results
    timer.stop();
    timer.print();

    return (0);
}




int Simulation :: compute_radiation_field_gpu ()
{
    const size_t nraypairs        =   0;     // Not used
    const size_t gpuBlockSize     = 512;
    const size_t gpuNumBlocks     =   1;     // Not used
    const double inverse_dtau_max = 100.0;   // Not used

    return gpu_compute_radiation_field (1, 1, 1, 1.0);
}




int Simulation :: gpu_compute_radiation_field (
        const size_t nraypairs,
        const size_t gpuBlockSize,
        const size_t gpuNumBlocks,
        const double inverse_dtau_max         )
{
    const size_t ncells = parameters.ncells();
    const size_t hnrays = parameters.nrays()/2;

    // Set timers
    Timer timer_overall("GPU compute radiation field -- overall");
    Timer timer_compute("GPU compute radiation field -- compute");

    timer_overall.start();

    timings.resize (hnrays);
    nrpairs.resize (hnrays);
    depths .resize (hnrays);

    for (size_t i = 0; i < hnrays; i++)
    {
        timings[i].resize (ncells);
        nrpairs[i].resize (ncells);
        depths [i].resize (ncells);
    }

    // Initialisations
    for (auto &lspec : lines.lineProducingSpecies) {lspec.lambda.clear ();}

    radiation.initialize_J ();

    /// Set maximum number of points along a ray, if not set yet
    if (geometry.max_npoints_on_rays == -1)
    {
        get_max_npoints_on_rays <CoMoving> ();
    }

    // Get number of threads
    const size_t nthreads = get_nthreads();

    size_t ngpu = 1;
    size_t ncpu;
    if (nthreads > ngpu) ncpu = nthreads - ngpu;
    else                 ncpu = 0;

    /// Create and initialize a solver for each thread
    vector<gpuSolver*> gpuSolvers (ngpu);
    vector<cpuSolver*> cpuSolvers (ncpu);

    for (auto &solver : gpuSolvers)
    {
        // Create a solver object
        solver = new gpuSolver (parameters.ncells(), parameters.nfreqs(),
                                parameters.nlines(), parameters.nboundary(),
                                parameters.ncells(), geometry.max_npoints_on_rays,
                                parameters.n_off_diag);

        /// Set GPU block size
        solver->gpuBlockSize     = gpuBlockSize;
        solver->gpuNumBlocks     = gpuNumBlocks;

        /// Set model data
        solver->copy_model_data (*this);
    }

    for (auto &solver : cpuSolvers)
    {
        // Create a solver object
        solver = new cpuSolver (parameters.ncells(), parameters.nfreqs(),
                                parameters.nlines(), parameters.nboundary(),
                                parameters.ncells(), geometry.max_npoints_on_rays,
                                parameters.n_off_diag);

        /// Set GPU block size
        solver->gpuBlockSize     = gpuBlockSize;
        solver->gpuNumBlocks     = gpuNumBlocks;

        /// Set model data
        solver->copy_model_data (*this);
    }


    for (size_t rr = 0; rr < hnrays; rr++)
    {
        logger.write ("ray = ", rr);

        const size_t RR = rr - MPI_start (hnrays);     // index ray
        const size_t ar = geometry.rays.antipod[rr];   // index antipod

        Queue queue (nraypairs);

        Size index = 0;

        #pragma omp parallel default (shared)
        {
            const Size thread_num = omp_get_thread_num();
            const bool gpu_thread = (thread_num < ngpu);

            while ((index < ncells) || !queue.queue.empty())
            {
                bool trace = false;
                Size o;

//                if (!gpu_thread)
                {
                    #pragma omp critical (update_queue)
                    {
                        trace = (index < ncells) && queue.is_not_full();

                        if (trace)
                        {
                            o = index++;
                        }
                    }
                }


                if (trace)
                {
                    const double dshift_max = get_dshift_max (o);

                    const RayData ray_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
                    const RayData ray_rr = geometry.trace_ray <CoMoving> (o, rr, dshift_max);

                    const size_t depth = ray_ar.size() + ray_rr.size() + 1;

                    if (depth > 1)
                    {
                        #pragma omp critical (update_queue)
                        {
                            queue.ordered_add (ray_ar, ray_rr, o, depth);
                        }
                    }
                    else
                    {
                        get_radiation_field_from_boundary (RR, rr, o);
                    }
                }
                else // solve
                {
                    bool       avail;   // True if a block is available to solve
                    ProtoBlock block;   // Available block to solve

                    #pragma omp critical (update_queue)
                    {
                        queue.get_block (gpu_thread, block, avail);
                    }

                    if (avail)
                    {
                        if (gpu_thread)
                        {
                            gpuSolver *solver = gpuSolvers[thread_num];

                            solver->nraypairs = block.nraypairs();
                            solver->width     = block.nraypairs() * parameters.nfreqs();

                            timer_compute.start();
                            solver->solve (block, RR, rr, *this);
                            timer_compute.stop();

                            timings[rr][solver->origins[0]] = solver->timer.get_interval();
                            nrpairs[rr][solver->origins[0]] = solver->nraypairs;
                            depths [rr][solver->origins[0]] = solver->n_tot[0];

                            #pragma omp critical (update_Lambda)
                            {
                                solver->update_Lambda (*this);
                            }
                        }
                        else
                        {
                            cpuSolver *solver = cpuSolvers[thread_num-ngpu];

                            solver->nraypairs = block.nraypairs();
                            solver->width     = block.nraypairs() * parameters.nfreqs();

                            timer_compute.start();
                            solver->solve (block, RR, rr, *this);
                            timer_compute.stop();

                            timings[rr][solver->origins[0]] = solver->timer.get_interval();
                            nrpairs[rr][solver->origins[0]] = solver->nraypairs;
                            depths [rr][solver->origins[0]] = solver->n_tot[0];

                            #pragma omp critical (update_Lambda)
                            {
                                solver->update_Lambda (*this);
                            }
                        }
                    }
                }
            }
        }
    }

    // Gather and reduce results of all MPI processes to get Lambda and J
#   if (MPI_PARALLEL)
        logger.write ("Gathering Lambda operators...");
        for (auto &lspec : lines.lineProducingSpecies) {lspec.lambda.MPI_gather ();}
        logger.write ("Reducing the mean intensities (J's)...");
        radiation.MPI_reduce_J ();
#   endif

    /// Delete solvers
    for (auto &solver : gpuSolvers) {delete solver;}
    for (auto &solver : cpuSolvers) {delete solver;}

    // Stop timer and print results
    timer_overall.stop();
    timer_compute.print();
    timer_overall.print();

    return (0);
}
