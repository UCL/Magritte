// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


//#include <Eigen/QR>
#include <limits>

#include "simulation.hpp"
#include "Tools/debug.hpp"
#include "Tools/logger.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/Parallel/hybrid.hpp"
#include "Functions/heapsort.hpp"
#include "Functions/planck.hpp"


///  Computer for spectral (=frequency) discretisation
//////////////////////////////////////////////////////

int Simulation ::
    compute_spectral_discretisation ()
{

  OMP_PARALLEL_FOR (p, parameters.ncells())
  {
    Double1 freqs (parameters.nfreqs());
    Long1   nmbrs (parameters.nfreqs());
    long    index0 = 0;
    long    index1 = 0;


    // Add the line frequencies (over the profile)
    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      const double inverse_mass = lines.lineProducingSpecies[l].linedata.inverse_mass;

      for (int k = 0; k < lines.lineProducingSpecies[l].linedata.nrad; k++)
      {
        const double freqs_line = lines.line[index0];
        const double width      = freqs_line * thermodynamics.profile_width (inverse_mass, p);

        for (long z = 0; z < parameters.nquads(); z++)
        {
          const double root = lines.lineProducingSpecies[l].quadrature.roots[z];

          freqs[index1] = freqs_line + width * root;
          nmbrs[index1] = index1;

          index1++;
        }

        index0++;
      }
    }
    /*
     *  Set other frequencies...
     */
    // // Add extra frequency bins around line to improve spectrum
    //
    // for (int l = 0; l < linedata.nlspec; l++)
    // {
    //   for (int k = 0; k < linedata.nrad[l]; k++)
    //   {
    //     const double freq_line = linedata.frequency[l][k];
    //     const double width     = profile_width (temperature.gas[p],
    //                                             temperature.vturb2[p],
    //                                             freq_line);
    //
    //     double factor = 1.0;
    //
    //     for (long e = 0; e < nbins; e++)
    //     {
    //       freqs[index1] = freq_line + width*LOWER * factor;
    //       nmbrs[index1] = index1;
    //
    //       index1++;
    //
    //       freqs[index1] = freq_line + width*UPPER * factor;
    //       nmbrs[index1] = index1;
    //
    //       index1++;
    //
    //       factor += 0.7;
    //     }
    //   }
    // }
    //
    //
    // // Add linspace for background
    //
    // // Find freqmax and freqmin
    //
    // long freqmax = 0;
    //
    // for (long f = 0; f < nfreq; f++)
    // {
    //   if (freqs[f] > freqmax)
    //   {
    //     freqmax = freqs[f];
    //   }
    // }
    //
    //
    // long freqmin = freqmax;
    //
    // for (long f = 0; f < nfreq; f++)
    // {
    //   if ( (freqs[f] < freqmin) && (freqs[f] != 0.0) )
    //   {
    //     freqmin = freqs[f];
    //   }
    // }
    //
    //
    // for (int i = 0; i < ncont; i++)
    // {
    //   freqs[index1] = (freqmax-freqmin) / ncont * i + freqmin;
    //   nmbrs[index1] = index1;
    //
    //   index1++;
    // }

    // Sort frequencies
    heapsort (freqs, nmbrs);


    // Set all frequencies nu
    for (long fl = 0; fl < parameters.nfreqs(); fl++)
    {
#     if (GRID_SIMD)
        const long    f = newIndex (fl);
        const long lane = laneNr   (fl);
        radiation.frequencies.nu[p][f].putlane (freqs[fl], lane);
#     else
        radiation.frequencies.nu[p][fl] = freqs[fl];
#     endif
    }


#   if (GRID_SIMD)

      // Remove possible first zeros

      for (int lane = n_simd_lanes-2; lane >= 0; lane--)
      {
        if (radiation.frequencies.nu[p][0].getlane(lane) <= 0.0)
        {
          const double freq = radiation.frequencies.nu[p][0].getlane(lane+1);

          radiation.frequencies.nu[p][0].putlane(0.9*freq, lane);
        }
      }

#   endif


    // Create lookup table for the frequency corresponding to each line
    Long1 nmbrs_inverted (parameters.nfreqs());

    for (long fl = 0; fl < parameters.nfreqs(); fl++)
    {
      nmbrs_inverted[nmbrs[fl]] = fl;

      radiation.frequencies.appears_in_line_integral[fl] = false;;
      radiation.frequencies.corresponding_l_for_spec[fl] = parameters.nfreqs();
      radiation.frequencies.corresponding_k_for_tran[fl] = parameters.nfreqs();
      radiation.frequencies.corresponding_z_for_line[fl] = parameters.nfreqs();
    }

    long index2 = 0;

    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      for (int k = 0; k < lines.lineProducingSpecies[l].nr_line[p].size(); k++)
      {
        for (long z = 0; z < lines.lineProducingSpecies[l].nr_line[p][k].size(); z++)
        {
          lines.lineProducingSpecies[l].nr_line[p][k][z] = nmbrs_inverted[index2];

          radiation.frequencies.appears_in_line_integral[index2] = true;
          radiation.frequencies.corresponding_l_for_spec[index2] = l;
          radiation.frequencies.corresponding_k_for_tran[index2] = k;
          radiation.frequencies.corresponding_z_for_line[index2] = z;

          index2++;
        }
      }
    }


    // Create lookup table for the transition corresponding to each frequency



  } // end of OMP_PARALLEL_FOR (p, parameters.ncells())


  return (0);

}




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

  for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
  {
    lspec.initialize_Lambda ();
  }

  radiation.initialize_J ();


  // Raypair along which the trasfer equation is solved
  RayPair rayPair;

  // Set bandwidth of the Approximated Lambda operator (ALO)
  rayPair.n_off_diag = parameters.n_off_diag;


  MPI_PARALLEL_FOR (r, parameters.nrays()/2)
  {
    const long R = r - MPI_start (parameters.nrays()/2);

    cout << "ray = " << r << endl;


#   pragma omp parallel default (shared) firstprivate (rayPair)
    {
    //OMP_FOR (o, parameters.ncells())

    // For better load balancing!!! (avoid getting boundary points on 1 thread)
    // removes any systematic in the distribution of points
    for (long o =  omp_get_thread_num(); o <  parameters.ncells(); o += omp_get_num_threads())
    {
      const long           ar = geometry.rays.antipod[o][r];
      const double weight_ang = geometry.rays.weights[o][r];
      const double dshift_max = get_dshift_max (o);


      // Trace and initialize the ray pair

      RayData rayData_r  = geometry.trace_ray <CoMoving> (o, r,  dshift_max);
      RayData rayData_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
      //cout << r << " " << o << " " << "   Rays traced!" << endl;

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

          //assert(false);
          //return (-1);

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


  // Reduce results of all MPI processes to get J and Lambda

# if (MPI_PARALLEL)

    radiation.MPI_reduce_J ();

    // "Reduce Lambda's"

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

  cout << "Creating an image along ray " << r << "..." << endl;

  // Create image object
  Image image (r, parameters);

  // Raypair along which the trasfer equation is solved
  RayPair rayPair;


  // if the ray is in this MPI process
  if (   (r >= MPI_start (parameters.nrays()/2))
      && (r <  MPI_stop  (parameters.nrays()/2)) )
  {
    const long R = r - MPI_start (parameters.nrays()/2);


#   pragma omp parallel default (shared) private (rayPair)
    {
    OMP_FOR (o, parameters.ncells())
    {
      const long           ar = geometry.rays.antipod[o][r];
      const double dshift_max = get_dshift_max (o);

      RayData rayData_r  = geometry.trace_ray <Rest> (o, r,  dshift_max);
      RayData rayData_ar = geometry.trace_ray <Rest> (o, ar, dshift_max);

      rayPair.initialize (rayData_ar.size(), rayData_r.size());

      if (rayPair.ndep > 1)
      {

        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          // Setup and solve the ray equations
          setup (R, o, f, rayData_ar, rayData_r, rayPair);

          rayPair.solve ();

          assert(false);

          // Store solution of the radiation field
          image.I_m[o][f] = rayPair.get_Im_at_front ();
          image.I_p[o][f] = rayPair.get_Ip_at_end ();
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







///  compute_LTE_level_populations: sets level populations to LTE values
////////////////////////////////////////////////////////////////////////

int Simulation ::
    compute_LTE_level_populations ()
{

  // Initialize levels, emissivities and opacities with LTE values
  lines.iteration_using_LTE (
      chemistry.species.abundance,
      thermodynamics.temperature.gas);


  return (0);

}




///  compute_level_populations: computes level populations self-consistenly with
///  the radiation field assuming statistical equilibrium
////////////////////////////////////////////////////////////////////////////////

int Simulation ::
    compute_level_populations (
        const Io   &io        )
{

  const bool use_Ng_acceleration = true;
  const long max_niterations     = 100;

  compute_level_populations_opts (io, use_Ng_acceleration, max_niterations);

  return (0);

}


///  compute_level_populations
//////////////////////////////

int Simulation ::
    compute_level_populations_opts (
        const Io   &io,
        const bool  use_Ng_acceleration,
        const long  max_niterations     )
{

  // Write out initial level populations
  for (int l = 0; l < parameters.nlspecs(); l++)
  {
    const string tag = "_iteration_0";

    lines.lineProducingSpecies[l].write_populations (io, l, tag);
  }

  // Initialize the number of iterations
  int iteration        = 0;
  int iteration_normal = 0;

  // Initialize errors
  error_mean.clear ();
  error_max.clear ();

  // Initialize some_not_converged
  bool some_not_converged = true;


  // Iterate as long as some levels are not converged
  while (some_not_converged && (iteration < max_niterations))
  {

    iteration++;

    cout << "Starting iteration " << iteration << endl;


    // Start assuming convergence
    some_not_converged = false;


    if (use_Ng_acceleration && (iteration_normal == 4))
    {
      lines.iteration_using_Ng_acceleration (
          parameters.pop_prec()             );

      iteration_normal = 0;
    }

    else
    {
      compute_radiation_field ();

      calc_Jeff ();

      lines.iteration_using_statistical_equilibrium (
          chemistry.species.abundance,
          thermodynamics.temperature.gas,
          parameters.pop_prec()                     );

      iteration_normal++;
    }


    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      error_mean.push_back (lines.lineProducingSpecies[l].relative_change_mean);
      error_max.push_back  (lines.lineProducingSpecies[l].relative_change_max);

      if (lines.lineProducingSpecies[l].fraction_not_converged > 0.005)
      {
        some_not_converged = true;
      }

      cout << "Already ";
      cout << 100 * (1.0 - lines.lineProducingSpecies[l].fraction_not_converged);
      cout << " % converged" << endl;

      const string tag = "_iteration_" + std::to_string (iteration);

      lines.lineProducingSpecies[l].write_populations (io, l, tag);
    }


  } // end of while loop of iterations


  // Print convergence stats
  cout << "Converged after " << iteration << " iterations" << endl;


  return (0);

}






///  calc_J_eff: calculate the effective mean intensity in a line
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
/////////////////////////////////////////////////////////////////////////////

void Simulation ::
    calc_Jeff ()
{

  for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
  {
    OMP_PARALLEL_FOR (p, parameters.ncells())
    {
      for (int k = 0; k < lspec.linedata.nrad; k++)
      {
        const Long1 freq_nrs = lspec.nr_line[p][k];

        lspec.Jeff[p][k] = 0.0;
        lspec.Jlin[p][k] = 0.0;


        // Integrate over the line

        for (long z = 0; z < parameters.nquads(); z++)
        {
          const double JJ = radiation.get_J (p,freq_nrs[z]);

          lspec.Jeff[p][k] += lspec.quadrature.weights[z] * JJ;
          lspec.Jlin[p][k] += lspec.quadrature.weights[z] * JJ;
        }


        // Subtract the approximated part

        for (long m = 0; m < lspec.lambda[p][k].nr.size(); m++)
        {
          const long I = lspec.index (lspec.lambda[p][k].nr[m], lspec.linedata.irad[k]);

          lspec.Jeff[p][k] -= HH_OVER_FOUR_PI * lspec.lambda[p][k].Ls[m] * lspec.population [I];
        }
      }
    }
  }


}




// ///  Computer for the number of points on each ray
// //////////////////////////////////////////////////
//
// int Simulation ::
//     compute_number_of_points_on_rays ()
// {
//
//   // Initialisation
//
//   Long2 npoints (parameters.nrays(), Long1 (parameters.ncells()));
//
//
//   MPI_PARALLEL_FOR (r, parameters.nrays()/2)
//   {
//     cout << "ray = " << r << endl;
//
// #   pragma omp parallel default (shared)
//     {
//     OMP_FOR (o, parameters.ncells())
//     {
//       const long           ar = geometry.rays.antipod[o][r];
//       const double dshift_max = get_dshift_max (o);
//
//
//       // Trace and initialize the ray pair
//
//       RayData rayData_r  = geometry.trace_ray <CoMoving> (o, r,  dshift_max);
//       RayData rayData_ar = geometry.trace_ray <CoMoving> (o, ar, dshift_max);
//
//       npoints[r][o] = rayData_r.size() + rayData_ar.size() + 1;
//
//     } // end of loop over cells
//     }
//
//   } // end of loop over ray pairs
//
//
//   return (0);
//
// }
