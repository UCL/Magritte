// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/Parallel/wrap_Grid.hpp"


///  Getter for the maximum allowed shift value determined by the smallest line
///    @param[in] o : number of point under consideration
///    @retrun maximum allowed shift value determined by the smallest line
///////////////////////////////////////////////////////////////////////////////

inline double Simulation ::
    get_dshift_max (
        const long o) const
{

  double dshift_max = std::numeric_limits<double>::max();   // Initialize to "infinity"

  for (const LineProducingSpecies &lspec : lines.lineProducingSpecies)
  {
    const double inverse_mass   = lspec.linedata.inverse_mass;
    const double new_dshift_max = parameters.max_width_fraction
                                  * thermodynamics.profile_width (inverse_mass, o);

    if (dshift_max > new_dshift_max)
    {
      dshift_max = new_dshift_max;
    }
  }


  return dshift_max;

}




///  Setup a ray pair for solving (in case of scattering)
///    @param[in]  R          : (local) index of the ray
///    @param[in]  origin     : index of the cell
///    @param[in]  f          : index of the frequency bin
///    @param[in]  rayData_r  : data along the ray r
///    @param[in]  rayData_ar : data along the (antipodal) ray ar
///    @param[out] rayPair    : raypair to set up
/////////////////////////////////////////////////////////////////

inline void Simulation ::
    setup_using_scattering (
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

  vReal freq_scaled = radiation.frequencies.nu[origin][f];
  long       cellNr = origin;
  long        index = rayData_ar.size();

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

      radiation.rescale_U_and_V (freq_scaled, R, data.cellNr, data.notch, U_scaled, V_scaled);

      get_eta_and_chi (freq_scaled, data.cellNr, data.lnotch, eta, chi);

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

      radiation.rescale_U_and_V (freq_scaled, R, data.cellNr, data.notch, U_scaled, V_scaled);

      get_eta_and_chi (freq_scaled, data.cellNr, data.lnotch, eta, chi);

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




///  Setup a ray pair for solving (in case of no scattering)
///    @param[in]  R          : (local) index of the ray
///    @param[in]  origin     : index of the cell
///    @param[in]  f          : index of the frequency bin
///    @param[in]  rayData_r  : data along the ray r
///    @param[in]  rayData_ar : data along the (antipodal) ray ar
///    @param[out] rayPair    : raypair to set up
/////////////////////////////////////////////////////////////////

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

  //vReal tau_r  = 0.0;
  //vReal tau_ar = 0.0;

  rayPair.first = 0;
  rayPair.last  = rayPair.n_ar;


  // Set origin

  vReal freq_scaled = radiation.frequencies.nu[origin][f];
  long       cellNr = origin;
  long        index = rayData_ar.size();


  get_eta_and_chi (freq_scaled, cellNr, rayPair.lnotch_at_origin, eta, chi);

  rayPair.set_term1_and_term2 (eta, chi, index);

  rayPair.chi[index] = chi;
  rayPair.nrs[index] = cellNr;
  rayPair.frs[index] = freq_scaled;

  vReal chi_prev = chi;
  vReal chi_o    = chi;


  // Set ray ar

  if (rayData_ar.size() > 0)
  {
    index = rayData_ar.size() - 1;

    for (ProjectedCellData &data : rayData_ar)
    {
      freq_scaled = data.shift * radiation.frequencies.nu[origin][f];

      //cout << "Getting eta and chi " << data.lnotch << endl;

      get_eta_and_chi (freq_scaled, data.cellNr, data.lnotch, eta, chi);

      //cout << "Got eta and chi     " << data.lnotch << endl;
      rayPair.set_term1_and_term2 (eta, chi,               index);
      rayPair.set_dtau            (chi, chi_prev, data.dZ, index);
      //cout << "Set terms and dtau " << endl;

      //tau_ar += rayPair.dtau[index];

      rayPair.chi[index] = chi;
      rayPair.nrs[index] = data.cellNr;
      rayPair.frs[index] = freq_scaled;

      chi_prev = chi;
      index--;

      //if (all_greather_then (tau_ar, tau_max)) break;
    }

    //if (geometry.boundary.boundary[rayPair.nrs[index+1]])
    //{
      cellNr = rayData_ar.back().cellNr;
       notch = rayData_ar.back().notch;
       bdyNr = geometry.boundary.cell2boundary_nr[cellNr];

      radiation.rescale_I_bdy (freq_scaled, R, cellNr, bdyNr, notch, rayPair.I_bdy_0);
    //}
    //else
    //{
    //  rayPair.I_bdy_0 = 0.0;
    //}

    rayPair.first = index+1;

    chi_prev = chi_o;
  }

  else
  {
    bdyNr = geometry.boundary.cell2boundary_nr[origin];

    rayPair.I_bdy_0 = radiation.I_bdy[R][bdyNr][f];
  }
  //cout << "Not in first part" << endl;


  // Set ray r

  if (rayData_r.size() > 0)
  {
    index = rayData_ar.size() + 1;

    for (ProjectedCellData &data : rayData_r)
    {
      freq_scaled = data.shift * radiation.frequencies.nu[origin][f];

      get_eta_and_chi (freq_scaled, data.cellNr, data.lnotch, eta, chi);

      rayPair.set_term1_and_term2 (eta, chi,               index  );
      rayPair.set_dtau            (chi, chi_prev, data.dZ, index-1);

      //tau_r += rayPair.dtau[index-1];

      rayPair.chi[index] = chi;
      rayPair.nrs[index] = data.cellNr;
      rayPair.frs[index] = freq_scaled;

      chi_prev = chi;
      index++;

      //if (all_greather_then (tau_r, tau_max)) break;
    }

    //if (geometry.boundary.boundary[rayPair.nrs[index-1]])
    //{
      cellNr = rayData_r.back().cellNr;
       notch = rayData_r.back().notch;
       bdyNr = geometry.boundary.cell2boundary_nr[cellNr];

      radiation.rescale_I_bdy (freq_scaled, R, cellNr, bdyNr, notch, rayPair.I_bdy_n);
    //}
    //else
    //{
    //  rayPair.I_bdy_n = 0.0;
    //}

    rayPair.last = index-1;
  }

  else
  {
    bdyNr = geometry.boundary.cell2boundary_nr[origin];

    rayPair.I_bdy_n = radiation.I_bdy[R][bdyNr][f];
  }

  //cout << "tau_r  = " << tau_r  << endl;
  //cout << "tau_ar = " << tau_ar << endl;

}




///  Getter for the line width
///    @param[in] p      : index of the cell
///    @paran[in] lindex : index of the line frequency
//////////////////////////////////////////////////////

inline double Simulation ::
    get_line_width (
        const long p,
        const long lindex) const
{
  const long           f = lines.line_index[lindex];
  const long           l = radiation.frequencies.corresponding_l_for_spec[f];
  const double invr_mass = lines.lineProducingSpecies[l].linedata.inverse_mass;

  return lines.line[lindex] * thermodynamics.profile_width (invr_mass, p);
}



///  Getter for the emissivity (eta) and the opacity (chi)
///    @param[in]     freq_scaled : frequency (in co-moving frame)
///    @param[in]     p           : in dex of the cell
///    @param[in/out] lnotch      : notch indicating location in the set of lines
///    @param[out]    eta         : emissivity
///    @param[out]    chi         : opacity
/////////////////////////////////////////////////////////////////////////////////

inline void Simulation ::
    get_eta_and_chi (
        const vReal &freq_scaled,
        const long   p,
              long  &lnotch,
              vReal &eta,
              vReal &chi         ) const
{
  //lnotch = 0;

  // Reset eta and chi
  eta = 0.0;
  chi = 1.0E-26;


  // GET LINE EMISIVITY AND OPACITY
  //_______________________________

  const double lower = -11.0; //1.1*lines.lineProducingSpecies[0].quadrature.roots[0];
  const double upper = +11.0; //1.1*lines.lineProducingSpecies[0].quadrature.roots[parameters.nquads()-1];


  // Move notch up to first line to include

  vReal  freq_diff = freq_scaled - (vReal) lines.line[lnotch];
  double     width = get_line_width (p, lnotch);


  //cout << "1. Starting from lnotch = " << lnotch << "   " << freq_diff << "   " << upper*width << endl;

  while (firstLane (freq_diff) > upper*width)
  {
    if (lnotch >= parameters.nlines()-1) break;

    lnotch++;

    freq_diff = freq_scaled - (vReal) lines.line[lnotch];
        width = get_line_width (p, lnotch);
  }

  //cout << "2. Stopped    at lnotch = " << lnotch << "   " << freq_diff << "   " << lower*width << endl;

  // Include all lines as long as frequency is close enough

  long lindex = lnotch;

  while (lastLane (freq_diff) >= lower*width)
  {
    const vReal line_profile = thermodynamics.profile (width, freq_diff);
    const long           ind = lines.index (p, lines.line_index[lindex]);

    eta += lines.emissivity[ind] * freq_scaled * line_profile;
    chi +=    lines.opacity[ind] * freq_scaled * line_profile;


    if (lindex >= parameters.nlines()-1) break;

    lindex++;

    freq_diff = freq_scaled - (vReal) lines.line[lindex];
        width = get_line_width (p, lindex);
  }

  //cout << "3. Last added    lindex = " << lindex << endl;
  //cout << "-----------------------------------------------" << endl;


    // SET MINIMAL OPACITY (to avoid zero optical depth increments (dtau))
    //____________________________________________________________________

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




///  Getter for the number of points on a ray pair in each point
///    @param[in] frame : frame of reference (for velocities)
///    ! Frame is required since different reference frames yield
///      different interpolations of the velocities and hence
///      differnet numbers of points along the ray pair
/////////////////////////////////////////////////////////////////

template <Frame frame>
Long1 Simulation ::
    get_npoints_on_ray (
        const long r   ) const
{

  Long1 npoints (parameters.ncells ());


  OMP_FOR (o, parameters.ncells ())
  {
    const long           ar = geometry.rays.antipod[o][r];
    const double dshift_max = get_dshift_max (o);

    RayData rayData_r  = geometry.trace_ray <frame> (o, r,  dshift_max);
    RayData rayData_ar = geometry.trace_ray <frame> (o, ar, dshift_max);

    npoints[o] = rayData_ar.size() + rayData_r.size() + 1;
  }


  return npoints;

}




///  Getter for the maximum number of points on a ray pair
///    @param[in] frame : frame of reference (for velocities)
///    ! Frame is required since different reference frames yield
///      different interpolations of the velocities and hence
///      differnet numbers of points along the ray pair
/////////////////////////////////////////////////////////////////

template <Frame frame>
long Simulation ::
    get_max_npoints_on_ray (
        const long r       ) const
{

  const Long1 npoints_on_ray = get_npoints_on_ray <frame> (r);

  return *std::max_element (npoints_on_ray.begin(), npoints_on_ray.end());

}




///  Getter for the number of points on each ray pair in each point
///    @param[in] frame : frame of reference (for velocities)
///    ! Frame is required since different reference frames yield
///      different interpolations of the velocities and hence
///      differnet numbers of points along the ray pair
///////////////////////////////////////////////////////////////////

template <Frame frame>
Long2 Simulation ::
    get_npoints_on_rays () const
{

  Long2 npoints (parameters.nrays()/2);


  MPI_PARALLEL_FOR (r, parameters.nrays()/2)
  {
    npoints[r] = get_max_npoints_on_ray <frame> (r);
  }


  return npoints;

}




///  Getter for the maximum number of points on a ray pair
///    @param[in] frame : frame of reference (for velocities)
///    ! Frame is required since different reference frames yield
///      different interpolations of the velocities and hence
///      differnet numbers of points along the ray pair
///////////////////////////////////////////////////////////////////

template <Frame frame>
long Simulation ::
    get_max_npoints_on_rays ()
{

  long  maximum = 0;


  MPI_PARALLEL_FOR (r, parameters.nrays()/2)
  {
    const long local_max = get_max_npoints_on_ray <frame> (r);

    if (maximum < local_max) maximum = local_max;
  }


  // Set max_npoints_on_rays in geometry
  geometry.max_npoints_on_rays = maximum;


  return maximum;

}
