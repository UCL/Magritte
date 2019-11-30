#include "Simulation/simulation.hpp"
#include "Raypair/raypair.cuh"



int Simulation :: gpu_get_device_properties (void)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  for (int i = 0; i < nDevices; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    const double pmb = 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;

    printf("------------------------------------\n"                        );
    printf("Device Number                : %d   \n", i                     );
    printf("Device name                  : %s   \n", prop.name             );
    printf("Compute compatibility        : %d.%d\n", prop.major, prop.minor);
    printf("Memory Clock Rate (KHz)      : %d   \n", prop.memoryClockRate  );
    printf("Memory Bus Width (bits)      : %d   \n", prop.memoryBusWidth   );
    printf("Peak Memory Bandwidth (GB/s) : %f   \n", pmb                   );
    printf("------------------------------------\n"                        );
  }

  return (0);
}


int Simulation :: gpu_compute_radiation_field (void)
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
  gpuRayPair *raypair = new gpuRayPair (geometry.max_npoints_on_rays,
                                        parameters.ncells(),
                                        parameters.nfreqs(),
                                        parameters.nlines()          );

  /// Set model data
  raypair->copy_model_data (Simulation(*this));


  for (long r = 0; r < parameters.nrays()/2; r++)
  {
    const long R = r - MPI_start (parameters.nrays()/2);

    logger.write ("ray = ", r);

    for (long o = 0; o < parameters.ncells(); o++)
    {
      const long           ar = geometry.rays.antipod[o][r];
      const double weight_ang = geometry.rays.weights[o][r];
      const double dshift_max = get_dshift_max (o);


      // Trace ray pair
      const RayData raydata_r  = geometry.trace_ray <CoMoving> (o, r,  dshift_max);
      const RayData raydata_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);

      if (raydata_r.size() + raydata_ar.size() > 0)
      {
        /// Setup such that the first ray is the longest (for performance)
        raypair->setup (*this, raydata_ar, raydata_r, R, o);
        /// Solve radiative transfer along ray pair
        raypair->solve ();
        /// Extract model data
        raypair->extract_radiation_field (*this, R, r, o);
      }
      else
      {
        /// Extract radiation field from boundary consitions
        get_radiation_field_from_boundary (R, r, o);
      }
    }
  }

  /// Delete raypair
  delete raypair;

  return (0);
}
