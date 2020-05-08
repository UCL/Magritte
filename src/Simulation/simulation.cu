#include "Simulation/simulation.hpp"
#include "Raypair/raypair.cuh"
#include "Raypair/rayblock.cuh"


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




