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


#include "Raypair/rayblock.hpp"


int Simulation :: cpu_compute_radiation_field_2 (const size_t nraypairs)
{
    // Initialisations
    for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
    {
        lspec.lambda.clear ();
    }

    radiation.initialize_J ();

    /// Set maximum number of points along a ray, if not set yet
    if (geometry.max_npoints_on_rays == -1)
    {
        get_max_npoints_on_rays <CoMoving> ();
    }

    /// Create a gpuRayPair object
    RayBlock *rayblock = new RayBlock (parameters.ncells(),
                                       parameters.nfreqs(),
                                       parameters.nlines(),
                                       nraypairs,
                                       geometry.max_npoints_on_rays);

    /// Set model data
    rayblock->copy_model_data (*this);


    for (size_t rr = 0; rr < parameters.nrays()/2; rr++)
    {
        const size_t         RR = rr - MPI_start (parameters.nrays()/2);
        const size_t         ar = geometry.rays.antipod[rr];
        const double weight_ang = geometry.rays.weights[rr];

        RayQueue rayqueue (nraypairs);


        logger.write ("ray = ", rr);

        for (size_t o = 0; o < parameters.ncells(); o++)
        {
            const double dshift_max = get_dshift_max (o);

            // Trace ray pair
            const RayData ray_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
            const RayData ray_rr = geometry.trace_ray <CoMoving> (o, rr, dshift_max);

            const size_t depth = ray_ar.size() + ray_rr.size() + 1;

            if (depth > 1)
            {
                /// Add ray pair to queue
                rayqueue.add (ray_ar, ray_rr, o, depth);

                if (rayqueue.complete)
                {
                    rayblock->setup (*this, RR, rr, rayqueue.get_complete_block());
                    rayblock->solve ();
                    rayblock->store (*this);
                }
            }
            else
            {
                /// Extract radiation field from boundary
                get_radiation_field_from_boundary (RR, rr, o);
            }
        }

        /// Compute the unfinished rays in the queue
        long s = 0;
        for (const ProtoRayBlock &prb : rayqueue.queue)
        {
            rayblock->nraypairs = prb.nraypairs();
            rayblock->width     = prb.nraypairs() * parameters.nfreqs();

            rayblock->setup (*this, RR, rr, prb);
            rayblock->solve ();
            rayblock->store (*this);

            cout << s << ":  o = " << prb.origins[0] << ", nraypairs = " << prb.nraypairs() << endl;
            s++;
        }
    }

    /// Delete ray block
    delete rayblock;

    return (0);
}
