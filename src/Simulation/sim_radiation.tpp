// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Image/image.hpp"
#include "Raypair/raypair.hpp"
#include "Functions/planck.hpp"


///  Computer for boundary intensities (setting the boundary conditions for RT)
///////////////////////////////////////////////////////////////////////////////

int Simulation ::
    compute_boundary_intensities ()
{

  for (long r = 0; r < parameters.nrays_red(); r++)
  {
    OMP_PARALLEL_FOR (b, parameters.nboundary())
    {
      const long p = geometry.boundary.boundary2cell_nr[b];

      for (long f = 0; f < parameters.nfreqs_red(); f++)
      {
        radiation.I_bdy[r][b][f] = planck (T_CMB, radiation.frequencies.nu[p][f]);
      }
    }
  }


  return (0);

}




///  Computer for the radiation field
/////////////////////////////////////

int Simulation ::
    compute_radiation_field ()
{

  // Initialisations
  cout << "Initializing..." << endl;

  for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
  {
    lspec.lambda.clear ();
  }

  radiation.initialize_J ();

  // Get the number of available threads
  cout << "Getting nthreads..." << endl;

  const int nthrds = get_nthreads ();

  cout << "nthrds = " << nthrds                << endl;
  cout << "ncells = " << parameters.ncells()   << endl;
  cout << "noff_d = " << parameters.n_off_diag << endl;


  if (geometry.max_npoints_on_rays == -1)
  {
    get_max_npoints_on_rays <CoMoving> ();
  }


  // Raypair along which the trasfer equation is solved
  cout << "Initializing raypair..." << endl;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// The resizing of the RayPair objects is left out of the RayPair constructor
//// since it (in some cases) yielded "General Protection Faults".
//// See Issue #6 in Grid-SIMD.
  if (nthrds > MAX_NTHREADS)
  {
    logger.write ("ERROR !!!                                                ");
    logger.write ("MAX_NTHREADS = ", (long) MAX_NTHREADS, " which is too low");
    logger.write ("(MAX_NTHREADS is defined in Tools/constants.hpp"          );
  }

  assert (nthrds <= MAX_NTHREADS);

  RayPair rayPairs [MAX_NTHREADS];

  for (int t = 0; t < nthrds; t++)
  {
    rayPairs[t].resize (geometry.max_npoints_on_rays, parameters.n_off_diag);
  }
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  cout << "Constructed RayPair array" << endl;


  // // Raypair along which the trasfer equation is solved
  // RayPair rayPair;

  // // Set bandwidth of the Approximated Lambda operator (ALO)
  // rayPair.n_off_diag = parameters.n_off_diag;


  MPI_PARALLEL_FOR (r, parameters.nrays()/2)
  {
    const long R = r - MPI_start (parameters.nrays()/2);

    logger.write ("ray = ", r);


#   pragma omp parallel default (shared)
    {
    // Create a reference to the ray pair object for this thread.
    // Required to avoid calls to the Grid-SIMD allocator (AllignedAllocator)
    // inside of an OpenMP (omp) parallel region.
    RayPair &rayPair = rayPairs[omp_get_thread_num()];

    //OMP_FOR (o, parameters.ncells())

    // For better load balancing!!! (avoid getting boundary points on 1 thread)
    // removes any systematic in the distribution of points
    for (long o = omp_get_thread_num(); o < parameters.ncells(); o += omp_get_num_threads())
    {
      const long           ar = geometry.rays.antipod[o][r];
      const double weight_ang = geometry.rays.weights[o][r];
      const double dshift_max = get_dshift_max (o);


      // Trace and initialize the ray pair

      //cout << "  " << r << "   " << o << endl;
      //cout << "  Try ray tracing!" << endl;
      RayData rayData_r  = geometry.trace_ray <CoMoving> (o, r,  dshift_max);
      RayData rayData_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
      //cout << "  Rays traced!" << endl;
      //cout << "  "<< rayData_r.size() << "   " << rayData_ar.size() << endl;

      if (rayData_r.size() > rayData_ar.size())
      {
        rayPair.initialize (rayData_r.size(), rayData_ar.size());
      }
      else
      {
        rayPair.initialize (rayData_ar.size(), rayData_r.size());
      }
      //cout << r << " " << o << " " << "   Raypair initialized!" << endl;



      // Solve radiative transfer along ray pair

      if (rayPair.ndep > 1)
      {
        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          //cout << "o = " << o << "   f = " << f << endl;

          // Setup and solve the ray equations

          //cout << "    setup" << endl;
          if (rayData_r.size() > rayData_ar.size())
          {
            //cout << "Inverted!" << endl;
            setup (R, o, f, rayData_r, rayData_ar, rayPair);
          }
          else
          {
            //cout << "Regular!"  << endl;
            setup (R, o, f, rayData_ar, rayData_r, rayPair);
          }
          //cout << r << " " << o << " " << f << "   Raypair set up!" << endl;


          rayPair.solve ();
          //cout << "Raypair solved!" << endl;


          rayPair.update_Lambda (
              radiation.frequencies,
              thermodynamics,
              o,
              f,
              weight_ang,
              lines                 );
          //cout << r << " " << o << " " << f << "   Lambda operator updated!" << endl;


          // Store solution of the radiation field
          const vReal u = rayPair.get_u_at_origin ();
          const vReal v = rayPair.get_v_at_origin ();

          const long ind = radiation.index (o,f);

          radiation.J[ind] += 2.0 * weight_ang * u;


          //cout << "nu["<<o<<"]["<<f<<"] = "<<radiation.frequencies.nu[o][f]<<endl;
          //cout << " u["<<o<<"]["<<f<<"] = "<<u<<endl;

          if (parameters.use_scattering())
          {
            if (rayData_r.size() > rayData_ar.size())
            {
              radiation.u[R][ind] =  u;
              radiation.v[R][ind] = -v;
            }
            else
            {
              radiation.u[R][ind] =  u;
              radiation.v[R][ind] =  v;
            }
          }

        }
      }

      else
      {
        // Only 2 points on the ray

        const long b = geometry.boundary.cell2boundary_nr[o];

        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          const vReal u = 0.5 * (radiation.I_bdy[R][b][f] + radiation.I_bdy[R][b][f]);
          const vReal v = 0.5 * (radiation.I_bdy[R][b][f] - radiation.I_bdy[R][b][f]);

          const long ind = radiation.index (o,f);

          radiation.J[ind] += 2.0 * weight_ang * u;

          if (parameters.use_scattering())
          {
            radiation.u[R][ind] = u;
            radiation.v[R][ind] = v;
          }
        }
      }


    } // end of loop over cells
    }


  } // end of loop over ray pairs



  // Gather and reduce results of all MPI processes to get Lambda and J

# if (MPI_PARALLEL)

    logger.write ("Gathering Lambda operators...");

    for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
    {
      lspec.lambda.MPI_gather ();
    }


    logger.write ("Reducing the mean intensities (J's)...");

    radiation.MPI_reduce_J ();

# endif


  if (parameters.use_scattering())
  {
    radiation.calc_U_and_V ();
  }


  return (0);

}




///  Computer and writer for images
///    @param[in] io : io object used to write the images
///    @param[in] r  : number of the ray indicating the direction of the image
//////////////////////////////////////////////////////////////////////////////

int Simulation ::
    compute_and_write_image (
        const Io  &io,
        const long r        )
{

  // Check spectral discretisation setting

  if (specDiscSetting != ImageSet)
  {
    logger.write ("Error: Spectral discretisation was not set for Images!");

    return (-1);
  }


  // Compute and write images

  cout << "Creating an image along ray " << r << "..." << endl;

  // Create image object
  Image image (r, parameters);

  // Get the number of available threads
  const int nthrds = get_nthreads ();

  // No need for the Lambda operator here
  const long n_off_diag = 0;


  cout << "nthrds = " << nthrds                << endl;
  cout << "ncells = " << parameters.ncells()   << endl;
  cout << "noff_d = " << parameters.n_off_diag << endl;

  const long max_npoints_on_ray = get_max_npoints_on_ray <Rest> (r);

  cout << "max_npoints_on_rays = " << max_npoints_on_ray << endl;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//// The resizing of the RayPair objects is left out of the RayPair constructor
//// since it (in some cases) yielded "General Protection Faults".
//// See Issue #6 in Grid-SIMD.
  if (nthrds > MAX_NTHREADS)
  {
    logger.write ("ERROR !!!                                                ");
    logger.write ("MAX_NTHREADS = ", (long) MAX_NTHREADS, " which is too low");
    logger.write ("(MAX_NTHREADS is defined in Tools/constants.hpp"          );
  }

  assert (nthrds <= MAX_NTHREADS);

  RayPair rayPairs [MAX_NTHREADS];

  for (int t = 0; t < nthrds; t++)
  {
    rayPairs[t].resize (max_npoints_on_ray, n_off_diag);
  }
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


  // if the ray is in this MPI process
  if (   (r >= MPI_start (parameters.nrays()/2))
      && (r <  MPI_stop  (parameters.nrays()/2)) )
  {
    const long R = r - MPI_start (parameters.nrays()/2);


#   pragma omp parallel default (shared)
    {
    // Create a reference to the ray pair object for this thread.
    // Required to avoid calls to the Grid-SIMD allocator (AllignedAllocator)
    // inside of an OpenMP (omp) parallel region.
    RayPair &rayPair = rayPairs[omp_get_thread_num()];

    //for (long o = omp_get_thread_num(); o < parameters.ncells(); o += omp_get_num_threads())
    //OMP_FOR (o, parameters.ncells())
    for (long o = 0; o < parameters.ncells(); o++)
    {
      cout << o << endl;

      const long           ar = geometry.rays.antipod[o][r];
      const double dshift_max = get_dshift_max (o);

      RayData rayData_r  = geometry.trace_ray <Rest> (o, r,  dshift_max);
      RayData rayData_ar = geometry.trace_ray <Rest> (o, ar, dshift_max);

      rayPair.initialize (rayData_ar.size(), rayData_r.size());

      if (rayPair.ndep > 1)
      {

        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          rayPair.first = 0;
          rayPair.last  = rayPair.ndep-1;

          // Setup and solve the ray equations
          setup (R, o, f, rayData_ar, rayData_r, rayPair);

          // Solve to get the intensities along the ray
          rayPair.solve_for_image ();

          // Store solution of the radiation field
          image.I_m[o][f] = rayPair.get_Im_at_front (); // rayPair.get_Im ();
          image.I_p[o][f] = rayPair.get_Ip_at_end   (); // rayPair.get_Ip ();
        }
      }

      else
      {
        const long b = geometry.boundary.cell2boundary_nr[o];

        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          image.I_m[o][f] = radiation.I_bdy[b][ar][f];
          image.I_p[o][f] = radiation.I_bdy[b][ r][f];
        }
      }


    } // end of loop over cells
    }

  }


  image.set_coordinates (geometry);


  image.write (io);


  return (0);

}
