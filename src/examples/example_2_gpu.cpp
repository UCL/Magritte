
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
#include "Tools/timer.hpp"


/// Test bench for performance testing of the setup and solve functions
/// that compute the radiation field in Magritte.
///////////////////////////////////////////////////////////////////////

int main (int argc, char **argv)
{
  /// Create a logger
  Logger logger ("example_2_gpu_benchmark");

  /// Error if no model file was providad as argument
  if (argc != 2)
  {
    logger.write ("Please provide a model file as argument."); return (-1);
  }

  /// Store model name
  const string modelName = argv[1];

  logger.write_line (                                                   );
  logger.write      ("   Magritte   (GPU)"                              );
  logger.write_line (                                                   );
  logger.write      ("Performance tests for setup and solver functions.");
  logger.write      ("( Only for GPU and single node use, i.e. no MPI )");
  logger.write_line (                                                   );
  logger.write      ("Running model: " + modelName                      );
  logger.write_line (                                                   );

  /// Create timer instances
  Timer timer1 ("copyT");
  Timer timer2 ("trace");
  Timer timer3 ("setup");
  Timer timer4 ("solve");
  Timer timer5 ("copyF");

  //IoPython io ("hdf5", modelName);
  IoText io (modelName);

  Simulation simulation;

  /// Write gpu properties
  simulation.gpu_get_device_properties();

  /// Set parameters
  simulation.parameters.set_pop_prec       (1.0E-6);
  simulation.parameters.set_use_scattering (false);

  simulation.read (io);

  simulation.compute_spectral_discretisation ();
  simulation.compute_boundary_intensities    ();
  simulation.compute_LTE_level_populations   ();

  if (simulation.geometry.max_npoints_on_rays == -1)
  {
    simulation.get_max_npoints_on_rays <CoMoving> ();
  }

  /// Create a gpuRayPair object
  cout << "Creating raypair..." << endl;
  gpuRayPair *raypair = new gpuRayPair (simulation.geometry.max_npoints_on_rays,
                                        simulation.parameters.ncells(),
                                        simulation.parameters.nfreqs(),
                                        simulation.parameters.nlines()          );

  /// Copy model data
  cout << "Copying model data..." << endl;
  timer1.start();
  raypair->copy_model_data (simulation);
  timer1.stop();
  // timer1.print();

  for (long r = 0; r < simulation.parameters.nrays()/2; r++)
  {
    const long R = r - MPI_start (simulation.parameters.nrays()/2);

    logger.write ("ray = ", r);


    for (long o = 0; o < simulation.parameters.ncells(); o++)
    {
      const double dshift_max = simulation.get_dshift_max (o);
      const double weight_ang = simulation.geometry.rays.weights[o][r];
      const long           ar = simulation.geometry.rays.antipod[o][r];

      /// Trace ray pair (and time it)
      timer2.start();
      const RayData raydata_r  = simulation.geometry.trace_ray <CoMoving> (o, r,  dshift_max);
      const RayData raydata_ar = simulation.geometry.trace_ray <CoMoving> (o, ar, dshift_max);
      timer2.stop();
      // timer2.print();

      if (raydata_r.size() + raydata_ar.size() > 0)
      {
        /// Setup such that the first ray is the longest (for performance)
        timer3.start();
        raypair->setup (simulation, raydata_ar, raydata_r, R, o);
        timer3.stop();
        // timer3.print();
        /// Solve radiative transfer along ray pair
        timer4.start();
        raypair->solve();
        timer4.stop();
        // timer4.print();

        /// Extract model data
        timer5.start();
        raypair->extract_radiation_field (simulation, R, r, o);
        timer5.stop();
        // timer5.print();

  //        timer3.start();
  //        rayPair.update_Lambda (
  //            simulation.radiation.frequencies,
  //            simulation.thermodynamics,
  //            o,
  //            f,
  //            weight_ang,
  //            simulation.lines                 );
  //        timer3.stop();
  //        timer3.print();
      } // end of if ndep > 1
    } // end of loop over cells
  } // end of loop over ray pairs

  /// Delete raypair
  delete raypair;

  /// Write output
  simulation.write (io);

  /// Print final timers
  timer1.print_total();
  timer2.print_total();
  timer3.print_total();
  timer4.print_total();
  timer5.print_total();

  /// Write exit message
  logger.write ("--- Magritte example 2 GPU benchmark is done.");

  return (0);
}
