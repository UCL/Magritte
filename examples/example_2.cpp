
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

  if (argc != 2)
  {
    cout << "Please provide a model file as argument." << endl;
  }

  else
  {
    const string modelName = argv[1];

    cout << "-------------------------------------------------" << endl;
    cout << "___Magritte______________________________________" << endl;
    cout << "-------------------------------------------------" << endl;
    cout << "Performance tests for setup and solver functions." << endl;
    cout << "(Only for single node use, i.e. no MPI.)         " << endl;
    cout << "-------------------------------------------------" << endl;
    cout << "Running model: " << modelName                      << endl;
    cout << "-------------------------------------------------" << endl;


#   pragma omp parallel
    {
      if (omp_get_thread_num() == 0)
      {
        cout << "n_omp_threads = " << omp_get_num_threads () << endl;
        cout << "n_simd_lanes  = " << n_simd_lanes           << endl;
      }
    }


    //IoPython io ("hdf5", modelName);
    IoText io (modelName);

    Simulation simulation;

    simulation.parameters.set_pop_prec       (1.0E-6);
    simulation.parameters.set_use_scattering (false);

    simulation.read (io);

    simulation.compute_spectral_discretisation ();

    simulation.compute_boundary_intensities ();

    simulation.compute_LTE_level_populations ();

    // Get the number of available threads
    int nthrds = get_nthreads ();

    // Raypair along which the trasfer equation is solved
    vector<RayPair> rayPairs (nthrds, RayPair (simulation.parameters.ncells (),
                                               simulation.parameters.n_off_diag));


    MPI_PARALLEL_FOR (r, simulation.parameters.nrays()/2)
    {
      const long R = r - MPI_start (simulation.parameters.nrays()/2);

      cout << "ray = " << r << endl;


#     pragma omp parallel default (shared)
      {
      // Create a reference to the ray pair object for this thread.
      // Required to avoid calls to the Grid-SIMD allocator (AllignedAllocator)
      // inside of an OpenMP (omp) parallel region.
      RayPair &rayPair = rayPairs[omp_get_thread_num()];

      Timer timer0("trace");
      Timer timer1("setup");
      Timer timer2("solve");

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
        timer0.print();

        rayPair.initialize (rayData_ar.size(), rayData_r.size());


        // Solve radiative transfer along ray pair

        if (rayPair.ndep > 1)
        {
          for (long f = 0; f < simulation.parameters.nfreqs_red(); f++)
          {
            // Setup and solve the ray equations

            timer1.start();
              simulation.setup (R, o, f, rayData_ar, rayData_r, rayPair);
            timer1.stop();
            timer1.print();


            timer2.start();
              rayPair.solve ();
            timer2.stop();
            timer2.print();

            cout << "----------------------------------" << endl;
          }
        }

      } // end of loop over cells

      timer0.print_total();
      timer1.print_total();
      timer2.print_total();

      }


    } // end of loop over ray pairs


    cout << "Done." << endl;
  }


  return (0);

}
