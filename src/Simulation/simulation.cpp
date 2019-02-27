// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <Eigen/QR>

#include "simulation.hpp"
#include "Tools/debug.hpp"
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
    long    index1 = 0;


    // Add the line frequencies (over the profile)
    for (int t = 0; t < parameters.nlines(); t++)
    {
      const double freqs_line = lines.line[t];
      const double width      = freqs_line * thermodynamics.profile_width (p);

      for (long z = 0; z < parameters.nquads(); z++)
      {
        freqs[index1] = freqs_line + width*lines.quadrature_roots[z];
        nmbrs[index1] = index1;
        index1++;
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
    }

    long index2 = 0;

    for (int l = 0; l < lines.nr_line[p].size(); l++)
    {
      for (int k = 0; k < lines.nr_line[p][l].size(); k++)
      {
        for (long z = 0; z < lines.nr_line[p][l][k].size(); z++)
        {
          lines.nr_line[p][l][k][z] = nmbrs_inverted[index2];
          index2++;
        }
      }
    }

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




///  compute_radiation_field
////////////////////////////

int Simulation ::
    compute_radiation_field ()
{

  RayPair rayPair;


  MPI_PARALLEL_FOR (r, parameters.nrays()/2)
  {
    const long R  = r - MPI_start (parameters.nrays()/2);
    const long ar = geometry.rays.antipod[r];

    std::cout << "ray = " << r << "   with antipodal = " << ar << std::endl;

//#   pragma omp parallel default (shared) private (rayPair)
//    for (long o = OMP_start (parameters.ncells()); o < OMP_stop (parameters.ncells()); o++)
    for (long o = 0; o < parameters.ncells(); o++)
    {

      const double dshift_max = 0.5 * thermodynamics.profile_width (o);

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

          radiation.u      [R][ind] = rayPair.get_u_at_origin ();
          radiation.v      [R][ind] = rayPair.get_v_at_origin ();

          radiation.u_local[R][ind] = rayPair.get_u_local_at_origin ();
        }
      }

      else
      {
        const long b = geometry.boundary.cell2boundary_nr[o];

        for (long f = 0; f < parameters.nfreqs_red(); f++)
        {
          const long ind = radiation.index (o,f);

          radiation.u      [R][ind] = 0.5 * (radiation.I_bdy[R][b][f] + radiation.I_bdy[R][b][f]);
          radiation.v      [R][ind] = 0.5 * (radiation.I_bdy[R][b][f] - radiation.I_bdy[R][b][f]);

          radiation.u_local[R][ind] = 0.5 * (radiation.I_bdy[R][b][f] + radiation.I_bdy[R][b][f]);
        }
      }


    } // end of loop over cells

  } // end of loop over ray pairs


  // Reduce results of all MPI processes to get J, J_local, U and V

  radiation.calc_J_and_J_local ();
  radiation.calc_U_and_V       ();


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
  chi = 0.0;


  const double lower = 1.00001*lines.quadrature_roots[0];
  const double upper = 1.00001*lines.quadrature_roots[parameters.nquads()-1];


  // Move notch up to first line to include

  vReal freq_diff = freq_scaled - (vReal) lines.line[lnotch];
  double    width = lines.line[lnotch] * thermodynamics.profile_width (p);


  while ( (firstLane (freq_diff) > upper*width) && (lnotch < parameters.nlines()-1) )
  {
    lnotch++;

    freq_diff = freq_scaled - (vReal) lines.line[lnotch];
        width = lines.line[lnotch] * thermodynamics.profile_width (p);
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
        width = lines.line[lindex] * thermodynamics.profile_width (p);
  }


  // Set minimal opacity to avoid zero optical depth increments (dtau)

# if (GRID_SIMD)

    GRID_FOR_ALL_LANES (lane)
    {
      if (fabs (chi.getlane(lane)) < 1.0E-99)
      {
        chi.putlane(1.0E-99, lane);
        eta.putlane((eta / (chi * 1.0E+99)).getlane(lane), lane);

        //cout << "WARNING : Opacity reached lower bound (1.0E-99)" << endl;
      }
    }

# else

    if (fabs (chi) < 1.0E-99)
    {
      chi = 1.0E-99;
      eta = eta / (chi * 1.0E+99);

      //cout << "WARNING : Opacity reached lower bound (1.0E-99)" << endl;
    }

# endif


}




///  compute_LTE_level_populations
//////////////////////////////////

int Simulation ::
    compute_LTE_level_populations ()
{

  OMP_PARALLEL_FOR (p, parameters.ncells())
  {
    const double temperature = thermodynamics.temperature.gas[p];

    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      const double abundance_lspec = chemistry.species.abundance[p][lines.linedata[l].num];

      lines.set_LTE_level_populations (abundance_lspec, temperature, p, l);

      lines.set_emissivity_and_opacity (p, l);
    }
  }


  return (0);

}




///  compute_level_populations
//////////////////////////////

int Simulation ::
    compute_level_populations ()
{

  // Initialize levels, emissivities and opacities with LTE values
  compute_LTE_level_populations ();

  // Initialize the number of iterations
  int iteration = 1;

  // Initialize some_not_converged
  bool some_not_converged = true;


  // Iterate as long as some levels are not converged
  while (some_not_converged && (iteration <= parameters.max_iter()))
  {
    std::cout << "Starting iteration " << iteration << std::endl;


    if (iteration % 4 == 0)
    {
      lines.update_using_Ng_acceleration ();
    }


    compute_radiation_field ();


    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      lines.         not_converged[l] = true;
      lines.fraction_not_converged[l] = 0.0;
    }


    update_using_statistical_equilibrium ();


    some_not_converged = false;

    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      if (lines.fraction_not_converged[l] > 0.005)
      {
        some_not_converged = true;
      }

      std::cout << 100*lines.fraction_not_converged[l];
      std::cout << " % not (yet) converged..." << endl;
    }


    iteration++;

  } // end of while loop of iterations


  // Print convergence stats
  std::cout << "Converged after " << iteration << " iterations" << std::endl;


  return (0);

}




int Simulation ::
    update_using_statistical_equilibrium ()
{

  double error_max_  = 0.0;
  double error_mean_ = 0.0;

  HYBRID_PARALLEL_FOR (p, parameters.ncells())
  {
    for (int l = 0; l < parameters.nlspecs(); l++)
    {
      const long nlev = lines.linedata[l].nlev;

      lines.population_prev3[p][l] = lines.population_prev2[p][l];
      lines.population_prev2[p][l] = lines.population_prev1[p][l];
      lines.population_prev1[p][l] = lines.population[p][l];


      Eigen::MatrixXd R = get_transition_matrix (p, l);
      Eigen::MatrixXd M = R.transpose();
      Eigen::VectorXd y = Eigen::VectorXd::Zero (nlev);


      for (long i = 0; i < nlev; i++)
      {
        double R_i = 0.0;

        for (long j = 0; j < nlev; j++)
        {
          R_i += R(i,j);
        }

        M(i,i) -= R_i;
      }


      // Replace last row with conservation equation

      for (long j = 0; j < nlev; j++)
      {
        M(nlev-1,j) = 1.0;
      }

      y(nlev-1) = lines.population_tot[p][l];


      // Solve matrix equation M*x=y for x
      lines.population[p][l] = M.householderQr().solve(y);


      // Avoid too small (or negative) populations

      for (int i = 0; i < nlev; i++)
      {
        if (lines.population[p][l](i)*lines.population_tot[p][l] < 1.0E-20)
        {
          lines.population[p][l](i) = 0.0;
        }
      }


      lines.set_emissivity_and_opacity (p, l);


      lines.check_for_convergence (p, l, parameters.pop_prec(), error_max_, error_mean_);
    }
  }

  error_max.push_back (error_max_);
  error_mean.push_back (error_mean_);


  // Gather emissivities and opacities from all processes

  lines.gather_emissivities_and_opacities ();


  return (0);

}




///  calc_J_and_L_eff: calculate the effective mean intensity in a line
///    @param[in] p: number of the cell under consideration
///    @param[in] l: number of the line producing species under consideration
/////////////////////////////////////////////////////////////////////////////

void Simulation ::
    calc_J_and_L_eff (
        const long p,
        const int  l,
        const long k )
{

  const Long1 freq_nrs = lines.nr_line[p][l][k];

  lines.J_line[p][l][k] = 0.0;
  lines.J_star[p][l][k] = 0.0;

  for (long z = 0; z < parameters.nquads(); z++)
  {
#   if (GRID_SIMD)
      const long    f = newIndex(freq_nrs[z]);
      const long lane =   laneNr(freq_nrs[z]);

      const double J  = radiation.J      [radiation.index(p,f)].getlane(lane);
      const double JL = radiation.J_local[radiation.index(p,f)].getlane(lane);
#   else
      const double J  = radiation.J      [radiation.index(p,freq_nrs[z])];
      const double JL = radiation.J_local[radiation.index(p,freq_nrs[z])];
#   endif

    lines.J_line[p][l][k] += lines.quadrature_weights[z] * J;
    lines.J_star[p][l][k] += lines.quadrature_weights[z] * JL;
  }


}



Eigen::MatrixXd Simulation ::
    get_transition_matrix (
        const long p,
        const long l      )
{

  Linedata linedata = lines.linedata[l];

  Eigen::MatrixXd R = Eigen::MatrixXd::Zero (linedata.nlev, linedata.nlev);


  // Radiative transitions

  for (int k = 0; k < linedata.nrad; k++)
  {
    const long i = linedata.irad[k];
    const long j = linedata.jrad[k];

    const long ind = lines.index (p, l, k);

    calc_J_and_L_eff (p, l, k);

    const double inverse_S_line = lines.opacity[ind] / lines.emissivity[ind];
    const double Jeff = lines.J_line[p][l][k] - lines.J_star[p][l][k];

    cout << lines.J_line[p][l][k] / lines.J_star[p][l][k] << endl;


    R(i,j) += linedata.A[k] * (1.0 - lines.J_star[p][l][k] * inverse_S_line);

    R(i,j) += linedata.Bs[k] * Jeff;
    R(j,i) += linedata.Ba[k] * Jeff;
  }


  // Collisional transitions

  for (CollisionPartner colpar : linedata.colpar)
  {
    double abn = chemistry.species.abundance[p][colpar.num_col_partner];
    double tmp = thermodynamics.temperature.gas[p];

    if (colpar.orth_or_para_H2 != "n")
    {
      const double frac_H2_para = 1.0 / (1.0 + 9.0*exp (-170.5/tmp));

      if (colpar.orth_or_para_H2 == "o")
      {
        abn *= (1.0 - frac_H2_para);
      }

      if (colpar.orth_or_para_H2 == "p")
      {
        abn *= frac_H2_para;
      }
    }


    colpar.interpolate_collision_coefficients (tmp);


    for (int k = 0; k < colpar.ncol; k++)
    {
      const int i = colpar.icol[k];
      const int j = colpar.jcol[k];

      R(i,j) += colpar.Cd_intpld[k] * abn;
      R(j,i) += colpar.Ce_intpld[k] * abn;
    }
  }


  return R;

}
