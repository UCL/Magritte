
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
  Logger logger ("example_2_cpu_benchmark");

  /// Error if no model file was providad as argument
  if (argc != 2)
  {
    cout << "Please provide a model file as argument." << endl;

    return (-1);
  }

  /// Store model name
  const string modelName = argv[1];

  logger.write ("-------------------------------------------------");
  logger.write ("   Magritte   (CPU)                              ");
  logger.write ("-------------------------------------------------");
  logger.write ("Performance tests for setup and solver functions.");
  logger.write ("(Only for CPU and single node use, i.e. no MPI.) ");
  logger.write ("-------------------------------------------------");
  logger.write ("Running model: " + modelName                      );
  logger.write ("-------------------------------------------------");

  /// Create timer instances
  Timer timer0("trace");
  Timer timer1("setup");
  Timer timer2("solve");
  Timer timer3("set_L");

  //IoPython io ("hdf5", modelName);
  IoText io (modelName);

  Simulation simulation;

  simulation.parameters.set_pop_prec       (1.0E-6);
  simulation.parameters.set_use_scattering (false);

  simulation.read (io);

  simulation.compute_spectral_discretisation ();
  simulation.compute_boundary_intensities    ();
  simulation.compute_LTE_level_populations   ();

  // Get the number of available threads
  const long nthrds = get_nthreads ();

  if (simulation.geometry.max_npoints_on_rays == -1)
  {
    simulation.get_max_npoints_on_rays <CoMoving> ();
  }

  // Raypair along which the trasfer equation is solved
  vector<RayPair> rayPairs (nthrds);

  // The resizing of the RayPair objects is left out of the RayPair constructor
  // since it (in some cases) yielded "General Protection Faults".
  // See Issue #6 in Grid-SIMD.
  for (int t = 0; t < rayPairs.size(); t++)
  {
    rayPairs[t].resize (simulation.geometry.max_npoints_on_rays, 0);
  }


  for (long r = 0; r < simulation.parameters.nrays()/2; r++)
  {
    const long R = r - MPI_start (simulation.parameters.nrays()/2);

    logger.write ("ray = ", r);


#    pragma omp parallel default (shared)
    {
    // Create a reference to the ray pair object for this thread.
    // Required to avoid calls to the Grid-SIMD allocator (AllignedAllocator)
    // inside of an OpenMP (omp) parallel region.
    RayPair &rayPair = rayPairs[omp_get_thread_num()];


    OMP_FOR (o, simulation.parameters.ncells())
    {
      const long           ar = simulation.geometry.rays.antipod[o][r];
      const double weight_ang = simulation.geometry.rays.weights[o][r];
      const double dshift_max = simulation.get_dshift_max (o);


      // Trace and initialize the ray pair

      timer0.start();
      RayData rayData_r  = simulation.geometry.trace_ray <CoMoving> (o, r,  dshift_max);
      RayData rayData_ar = simulation.geometry.trace_ray <CoMoving> (o, ar, dshift_max);
      timer0.stop();
      // timer0.print();

      if (rayData_r.size() > rayData_ar.size())
      {
        rayPair.initialize (rayData_r.size(), rayData_ar.size());
      }
      else
      {
        rayPair.initialize (rayData_ar.size(), rayData_r.size());
      }


      // Solve radiative transfer along ray pair

      if (rayPair.ndep > 1)
      {
        for (long f = 0; f < simulation.parameters.nfreqs_red(); f++)
        {
          // Setup and solve the ray equations

          timer1.start();
          if (rayData_r.size() > rayData_ar.size())
          {
            //cout << "Inverted!" << endl;
            simulation.setup (R, o, f, rayData_r, rayData_ar, rayPair);
          }
          else
          {
            //cout << "Regular!"  << endl;
            simulation.setup (R, o, f, rayData_ar, rayData_r, rayPair);
          }
          timer1.stop();
          // timer1.print();


          timer2.start();
            rayPair.solve ();
          timer2.stop();
          // timer2.print();


          timer3.start();
          rayPair.update_Lambda (
              simulation.radiation.frequencies,
              simulation.thermodynamics,
              o,
              f,
              weight_ang,
              simulation.lines                 );
          timer3.stop();
          // timer3.print();

          // Store solution of the radiation field
          const vReal u = rayPair.get_u_at_origin ();
          const vReal v = rayPair.get_v_at_origin ();

          const long ind = simulation.radiation.index (o,f);

          simulation.radiation.J[ind] += 2.0 * weight_ang * u;
        }
      }

    } // end of loop over cells
    }

  } // end of loop over ray pairs

  /// Write output
  simulation.write (io);

  /// Print final timers
  timer0.print_total();
  timer1.print_total();
  timer2.print_total();
  timer3.print_total();

  /// Print run specifications
  logger.write ("--- Magritte run specifications : ");
  logger.write ("n_omp_threads = ", nthrds          );
  logger.write ("n_sim_lanes   = ", n_simd_lanes    );

  /// Write exit message
  logger.write ("--- Magritte example 2 CPU benchmark is done.");

  return (0);
}
