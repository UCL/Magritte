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




///  get_dshift_max: the maximum allowed shift is determined by the smallest line
///    @param[in] o: number of point under consideration
/////////////////////////////////////////////////////////////////////////////////

inline double Simulation ::
    get_dshift_max (
        const long o)
{

  double dshift_max = std::numeric_limits<double>::max();   // Initialize to "infinity"

  for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
  {
    const double inverse_mass   = lspec.linedata.inverse_mass;
    const double new_dshift_max = 0.5 * thermodynamics.profile_width (inverse_mass, o);

    if (new_dshift_max < dshift_max)
    {
      dshift_max = new_dshift_max;
    }
  }

  return dshift_max;

}



///  compute_radiation_field
////////////////////////////

int Simulation ::
    compute_radiation_field ()
{


  for (LineProducingSpecies &lspec : lines.lineProducingSpecies)
  {
    lspec.initialize_Lambda ();
  }


  // Raypair along which the trasfer equation is solved
  RayPair rayPair;
  rayPair.n_off_diag = parameters.n_off_diag;
  cout << "Set n_off_diag " << rayPair.n_off_diag << endl;


  MPI_PARALLEL_FOR (r, parameters.nrays()/2)
  {
    const long R = r - MPI_start (parameters.nrays()/2);

    cout << "ray = " << r << endl;

#   pragma omp parallel default (shared) firstprivate (rayPair)
    {
    OMP_FOR (o, parameters.ncells())
    {
      const long           ar = geometry.rays.antipod[o][r];
      const double dshift_max = get_dshift_max (o);

      RayData rayData_r  = geometry.trace_ray (o, r,  dshift_max);
      RayData rayData_ar = geometry.trace_ray (o, ar, dshift_max);

      rayPair.initialize (rayData_ar.size(), rayData_r.size());


      if (rayPair.ndep > 1)
      {
        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          // Setup and solve the ray equations
          setup (R, o, f, rayData_ar, rayData_r, rayPair);
          rayPair.solve ();


          // Store solution of the radiation field
          const long ind = radiation.index (o,f);
          radiation.u[R][ind] = rayPair.get_u_at_origin ();
          radiation.v[R][ind] = rayPair.get_v_at_origin ();


          // Extract the Lambda operator
          rayPair.update_Lambda (
              radiation.frequencies,
              thermodynamics,
              o,
              f,
              geometry.rays.weights[o][r],
              lines                       );
        }
      }

      else
      {
        const long b = geometry.boundary.cell2boundary_nr[o];

        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          const long ind = radiation.index (o,f);

          radiation.u[R][ind] = 0.5 * (radiation.I_bdy[R][b][f] + radiation.I_bdy[R][b][f]);
          radiation.v[R][ind] = 0.5 * (radiation.I_bdy[R][b][f] - radiation.I_bdy[R][b][f]);
        }
      }


    } // end of loop over cells
    }

  } // end of loop over ray pairs


  // Reduce results of all MPI processes to get J, J_local, U and V

  radiation.calc_J_and_G (geometry.rays.weights);

  radiation.calc_U_and_V ();

  // "Reduce Lambda's"

  return (0);

}




///  compute_image
//////////////////

int Simulation ::
    compute_and_write_image (
        const Io  &io,
        const long r        )
{

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

      RayData rayData_r  = geometry.trace_ray (o, r,  dshift_max);
      RayData rayData_ar = geometry.trace_ray (o, ar, dshift_max);

      rayPair.initialize (rayData_ar.size(), rayData_r.size());


      if (rayPair.ndep > 1)
      {
        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          // Setup and solve the ray equations
          setup (R, o, f, rayData_ar, rayData_r, rayPair);
          rayPair.solve ();


          // Store solution of the radiation field
          image.I_m[o][f] = rayPair.get_I_m ();
          image.I_p[o][f] = rayPair.get_I_p ();
        }
      }

      else
      {
        const long b = geometry.boundary.cell2boundary_nr[o];

        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          image.I_m[o][f] = radiation.I_bdy[b][r][f];
          image.I_p[o][f] = radiation.I_bdy[b][r][f];
        }
      }

    } // end of loop over cells
    }

  }


  image.set_coordinates (geometry);


  image.write (io);


  return (0);

}



inline void Simulation ::
    setup (
        const long     R,
        const long     origin,
        const long     f,
              RayData &rayData_ar,
              RayData &rayData_r,
              RayPair &rayPair    ) const
{

  vReal eta;
  vReal chi;

  long notch;
  long bdyNr;


  // Set origin

  double freq_scaled = radiation.frequencies.nu[origin][f];
  long        cellNr = origin;
  long         index = rayData_ar.size();

  vReal U_scaled = radiation.get_U (R, origin, f);
  vReal V_scaled = radiation.get_V (R, origin, f);

  get_eta_and_chi (freq_scaled, cellNr, rayPair.lnotch_at_origin, eta, chi);

  rayPair.set_term1_and_term2 (eta, chi, U_scaled, V_scaled, index);

  rayPair.chi[index] = chi;
  rayPair.nrs[index] = cellNr;
  rayPair.frs[index] = freq_scaled;

  vReal chi_prev = chi;
  vReal chi_o    = chi;


  // Set ray ar

  if (rayData_ar.size() > 0)
  {
    index = rayData_ar.size() - 1;

    for (ProjectedCellData data : rayData_ar)
    {
      freq_scaled = data.shift * radiation.frequencies.nu[origin][f];

      radiation.rescale_U_and_V (freq_scaled, R, data.cellNr, data.notch,  U_scaled, V_scaled);
                get_eta_and_chi (freq_scaled,    data.cellNr, data.lnotch, eta,      chi     );

      rayPair.set_term1_and_term2 (eta, chi, U_scaled, V_scaled, index);
      rayPair.set_dtau            (chi, chi_prev, data.dZ,       index);

      rayPair.chi[index] = chi;
      rayPair.nrs[index] = data.cellNr;
      rayPair.frs[index] = freq_scaled;

      chi_prev = chi;
      index--;
    }

    cellNr = rayData_ar.back().cellNr;
     notch = rayData_ar.back().notch;
     bdyNr = geometry.boundary.cell2boundary_nr[cellNr];

    radiation.rescale_I_bdy (freq_scaled, R, cellNr, bdyNr, notch, rayPair.I_bdy_0);

    chi_prev = chi_o;
  }

  else
  {
    bdyNr = geometry.boundary.cell2boundary_nr[origin];

    rayPair.I_bdy_0 = radiation.I_bdy[R][bdyNr][f];
  }


  // Set ray r

  if (rayData_r.size() > 0)
  {
    index = rayData_ar.size() + 1;

    for (ProjectedCellData data : rayData_r)
    {
      freq_scaled = data.shift * radiation.frequencies.nu[origin][f];

      radiation.rescale_U_and_V (freq_scaled, R, data.cellNr, data.notch,  U_scaled, V_scaled);
                get_eta_and_chi (freq_scaled,    data.cellNr, data.lnotch, eta,      chi     );

      rayPair.set_term1_and_term2 (eta, chi, U_scaled, V_scaled, index  );
      rayPair.set_dtau            (chi, chi_prev, data.dZ,       index-1);

      rayPair.chi[index] = chi;
      rayPair.nrs[index] = data.cellNr;
      rayPair.frs[index] = freq_scaled;

      chi_prev = chi;
      index++;
    }

    cellNr = rayData_r.back().cellNr;
     notch = rayData_r.back().notch;
     bdyNr = geometry.boundary.cell2boundary_nr[cellNr];

    radiation.rescale_I_bdy (freq_scaled, R, cellNr, bdyNr, notch, rayPair.I_bdy_n);
  }

  else
  {
    bdyNr = geometry.boundary.cell2boundary_nr[origin];

    rayPair.I_bdy_n = radiation.I_bdy[R][bdyNr][f];
  }




}




///  get_eta_and_chi
////////////////////

inline void Simulation ::
    get_eta_and_chi (
        const vReal &freq_scaled,
        const long   p,
              long  &lnotch,
              vReal &eta,
              vReal &chi         ) const
{
  // TEMPORARY !!!

  lnotch = 0;

  ////////////////


  // Reset eta and chi
  eta = 0.0;
  chi = 1.0E-22;


  const double lower = 1.00001*lines.lineProducingSpecies[0].quadrature.roots[0];
  const double upper = 1.00001*lines.lineProducingSpecies[0].quadrature.roots[parameters.nquads()-1];


  // Move notch up to first line to include

  vReal  freq_diff = freq_scaled - (vReal) lines.line[lnotch];
  long           f = lines.line_index[lnotch];
  long           l = radiation.frequencies.corresponding_l_for_spec[f];
  double invr_mass = lines.lineProducingSpecies[l].linedata.inverse_mass;
  double    width = lines.line[lnotch] * thermodynamics.profile_width (invr_mass, p);


  while ( (firstLane (freq_diff) > upper*width) && (lnotch < parameters.nlines()-1) )
  {
    lnotch++;

    freq_diff = freq_scaled - (vReal) lines.line[lnotch];
            f = lines.line_index[lnotch];
            l = radiation.frequencies.corresponding_l_for_spec[f];
    invr_mass = lines.lineProducingSpecies[l].linedata.inverse_mass;
        width = lines.line[lnotch] * thermodynamics.profile_width (invr_mass, p);
  }


  // Include all lines as long as frequency is close enough

  long lindex = lnotch;

  while ( (lastLane (freq_diff) >= lower*width) && (lindex < parameters.nlines()) )
  {
    const vReal line_profile = thermodynamics.profile (width, freq_diff);
    const long           ind = lines.index (p, lines.line_index[lindex]);

    eta += lines.emissivity[ind] * freq_scaled * line_profile;
    chi +=    lines.opacity[ind] * freq_scaled * line_profile;

    lindex++;

    freq_diff = freq_scaled - (vReal) lines.line[lindex];
            f = lines.line_index[lnotch];
            l = radiation.frequencies.corresponding_l_for_spec[f];
    invr_mass = lines.lineProducingSpecies[l].linedata.inverse_mass;
        width = lines.line[lindex] * thermodynamics.profile_width (invr_mass, p);
  }


    // Set minimal opacity to avoid zero optical depth increments (dtau)

# if (GRID_SIMD)

    //GRID_FOR_ALL_LANES (lane)
    //{
    //  if (fabs (chi.getlane(lane)) < 1.0E-99)
    //  {
    //    eta.putlane((eta / (chi * 1.0E+99)).getlane(lane), lane);
    //    chi.putlane(1.0E-99, lane);

    //    //cout << "WARNING : Opacity reached lower bound (1.0E-99)" << endl;
    //  }
    //}

# else

  //  if (fabs (chi) < 1.0E-22)
  //  {
  //    eta = eta / (chi * 1.0E+22);
  //    chi = 1.0E-22;

      //cout << "WARNING : Opacity reached lower bound (1.0E-22)" << endl;
  //  }

# endif


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
#         if (GRID_SIMD)
            const long    f = newIndex (freq_nrs[z]);
            const long lane =   laneNr (freq_nrs[z]);

            const double JJ = radiation.J[radiation.index(p,f)].getlane(lane);
#         else
            const double JJ = radiation.J[radiation.index(p,freq_nrs[z])];
#         endif

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
