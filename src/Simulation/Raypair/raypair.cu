#include "raypair.cuh"
#include "Simulation/simulation.hpp"


///  Constructor for gpuRayPair
///    @param[in] mrl : maximum length of the ray pair max(n_ar + n_r + 1)
///    @param[in] ncs : number of cells
///    @param[in] nfs : number of frequency bins
///    @param[in] nls : number of lines
//////////////////////////////////////////////////////////////////////////

__host__
gpuRayPair ::
    gpuRayPair (
        const long mrl,
        const long ncs,
        const long nfs,
        const long nls )
    : maxLength (mrl)
    , ncells    (ncs)
    , nfreqs    (nfs)
    , nlines    (nls)
{
  cudaMallocManaged (&line,                         nlines*sizeof(double));
  cudaMallocManaged (&line_emissivity,       ncells*nlines*sizeof(double));
  cudaMallocManaged (&line_opacity,          ncells*nlines*sizeof(double));
  cudaMallocManaged (&width,                 ncells*nlines*sizeof(double));

  cudaMallocManaged (&freqs,              maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&freqs_scaled,       maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&freqs_lower,        maxLength*nfreqs*sizeof(long));
  cudaMallocManaged (&freqs_upper,        maxLength*nfreqs*sizeof(long));

  cudaMallocManaged (&I_bdy_0_presc,                nfreqs*sizeof(double));
  cudaMallocManaged (&I_bdy_n_presc,                nfreqs*sizeof(double));

  cudaMallocManaged (&term1,              maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&term2,              maxLength*nfreqs*sizeof(double));

  cudaMallocManaged (&eta,                maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&chi,                maxLength*nfreqs*sizeof(double));

  cudaMallocManaged (&nrs,                maxLength       *sizeof(long));
  cudaMallocManaged (&dZs,                maxLength       *sizeof(double));

  cudaMallocManaged (&A,                  maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&C,                  maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&F,                  maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&G,                  maxLength*nfreqs*sizeof(double));

  cudaMallocManaged (&inverse_A,          maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&inverse_C,          maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&inverse_one_plus_F, maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&inverse_one_plus_G, maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (& G_over_one_plus_G, maxLength*nfreqs*sizeof(double));

  cudaMallocManaged (&Su,                 maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&Sv,                 maxLength*nfreqs*sizeof(double));
  cudaMallocManaged (&dtau,               maxLength*nfreqs*sizeof(double));

  cudaMallocManaged (&L_diag,             maxLength*nfreqs*sizeof(double));

  return;
}


///  Destructor for gpuRayPair
//////////////////////////////

__host__
gpuRayPair ::
    ~gpuRayPair (void)
{
  cudaFree (line);
  cudaFree (line_emissivity);
  cudaFree (line_opacity);
  cudaFree (width);

  cudaFree (freqs);
  cudaFree (freqs_scaled);
  cudaFree (freqs_lower);
  cudaFree (freqs_upper);

  cudaFree (I_bdy_0_presc);
  cudaFree (I_bdy_n_presc);

  cudaFree (term1);
  cudaFree (term2);

  cudaFree (eta);
  cudaFree (chi);

  cudaFree (nrs);
  cudaFree (dZs);

  cudaFree (A);
  cudaFree (C);
  cudaFree (F);
  cudaFree (G);

  cudaFree (inverse_A);
  cudaFree (inverse_C);
  cudaFree (inverse_one_plus_F);
  cudaFree (inverse_one_plus_G);
  cudaFree ( G_over_one_plus_G);

  cudaFree (Su);
  cudaFree (Sv);
  cudaFree (dtau);

  cudaFree (L_diag);

  return;
}


///  Copies the model data into the gpuRayPair data structure
///    @param[in] model : model from which to copy
/////////////////////////////////////////////////////////////

int gpuRayPair :: copy_model_data (const Model &model)
{
  cudaMemcpy (line,
              model.lines.line.data(),
              model.lines.line.size()*sizeof(double),
              cudaMemcpyHostToDevice  );
  cudaMemcpy (line_emissivity,
              model.lines.emissivity.data(),
              model.lines.emissivity.size()*sizeof(double),
              cudaMemcpyHostToDevice  );
  cudaMemcpy (line_opacity,
              model.lines.opacity.data(),
              model.lines.opacity.size()*sizeof(double),
              cudaMemcpyHostToDevice  );


  for (long p = 0; p < ncells; p++)
  {
    for (long l = 0; l < nlines; l++)
    {
      const long           f = model.lines.line_index[l];
      const long       lspec = model.radiation.frequencies.corresponding_l_for_spec[f];
      const double invr_mass = model.lines.lineProducingSpecies[lspec].linedata.inverse_mass;

      width[L(p,l)] = model.thermodynamics.profile_width (invr_mass, p, model.lines.line[l]);
    }
  }

  return (0);
}


__host__
void gpuRayPair ::
    setFrequencies (
        const Double1 &frequencies,
        const double   scale,
        const long     index       )
{
  long f1 = 0;

  for (long f2 = 0; f2 < nfreqs; f2++)
  {
    const long if2 = I(index,f2);

    freqs       [if2] = frequencies[f2];
    freqs_scaled[if2] = frequencies[f2] * scale;

    while (   (frequencies[f1] < freqs_scaled[if2])
           && (f1              < nfreqs-1         ) )
    {
      f1++;
    }

    if (f1 > 0)
    {
      freqs_lower[if2] = f1-1;
      freqs_upper[if2] = f1;
    }
    else
    {
      freqs_lower[if2] = 0;
      freqs_upper[if2] = 1;
    }
  }
}


__host__
void gpuRayPair ::
    setup (
        const Model   &model,
        const RayData &raydata1,
        const RayData &raydata2,
        const long     R,
        const long     o          )
{
  /// For interchanging raydata
  RayData raydata_ar;
  RayData raydata_r;

  /// Ensure that ray ar is longer than ray r
  if (raydata2.size() > raydata1.size())
  {
    reverse    =     -1.0;
    raydata_ar = raydata2;
    raydata_r  = raydata1;
  }
  else
  {
    reverse    =     +1.0;
    raydata_ar = raydata1;
    raydata_r  = raydata2;
  }

  /// Extract ray and antipodal ray lengths
  n_ar = raydata_ar.size();
  n_r  = raydata_r. size();

  /// Set total number of depth points
  ndep = n_ar + n_r + 1;

  /// Initialize index
  long index = n_ar;

  /// Temporarily set first and last
  first = 0;
  last  = n_ar;

  /// Set origin
  setFrequencies (model.radiation.frequencies.nu[o], 1.0, index);
  nrs[index] = o;

  /// Temporary boundary numbers
  long bdy_0 = model.geometry.boundary.cell2boundary_nr[o];
  long bdy_n = model.geometry.boundary.cell2boundary_nr[o];

  /// Set ray ar
  if (n_ar > 0)
  {
    index = n_ar - 1;

    for (const ProjectedCellData &data : raydata_ar)
    {
      setFrequencies (model.radiation.frequencies.nu[o], data.shift, index);
      nrs[index] = data.cellNr;
      dZs[index] = data.dZ;
      index--;
    }

    bdy_0 = model.geometry.boundary.cell2boundary_nr[raydata_ar.back().cellNr];
    first = index+1;
  }

  /// Set ray r
  if (n_r > 0)
  {
    index = n_ar + 1;

    for (const ProjectedCellData &data : raydata_r)
    {
      setFrequencies (model.radiation.frequencies.nu[o], data.shift, index);
      nrs[index  ] = data.cellNr;
      dZs[index-1] = data.dZ;
      index++;
    }

    bdy_n = model.geometry.boundary.cell2boundary_nr[raydata_r.back().cellNr];
    last  = index-1;
  }

  /// Copy boundary intensities
  cudaMemcpy (I_bdy_0_presc,
              model.radiation.I_bdy[R][bdy_0].data(),
              model.radiation.I_bdy[R][bdy_0].size()*sizeof(double),
              cudaMemcpyHostToDevice                                );
  cudaMemcpy (I_bdy_n_presc,
              model.radiation.I_bdy[R][bdy_n].data(),
              model.radiation.I_bdy[R][bdy_n].size()*sizeof(double),
              cudaMemcpyHostToDevice                                );

  return;
}



__global__
void feautrierKernel (gpuRayPair &raypair)
{
  const long index  = blockIdx.x * blockDim.x + threadIdx.x;
  const long stride =  gridDim.x * blockDim.x;

  for (long f = index; f < raypair.nfreqs; f += stride)
  {
    raypair.solve_Feautrier (f);
  }

  return;
}


__host__
void gpuRayPair :: solve (void)
{
  const long blockSize = 32;
  const long numBlocks = (nfreqs + blockSize-1) / blockSize;

  feautrierKernel <<<numBlocks, blockSize>>> (*this);

  // Wait for GPU to finish and get possible error
  HANDLE_ERROR(cudaDeviceSynchronize());

  return;
}

__host__
void gpuRayPair :: extract_radiation_field (
          Model &model,
    const long   R,
    const long   r,
    const long   o                         )
{
  const double weight_ang = 2.0 * model.geometry.rays.weights[r];

  const long i0 = model.radiation.index(o,0);
  const long j0 = I(n_ar,0);

  for (long f = 0; f < nfreqs; f++)
  {
    model.radiation.J[i0+f] += weight_ang * Su[j0+f];
  }

  if (model.parameters.use_scattering())
  {
    for (long f = 0; f < nfreqs; f++)
    {
      model.radiation.u[R][i0+f] = Su[j0+f];
      model.radiation.v[R][i0+f] = Sv[j0+f] * reverse;
    }
  }

  return;
}



///  Gaussian line profile function
///    @param[in] width     : profile width
///    @param[in] freq_diff : frequency difference with line centre
///    @return profile function evaluated with this frequency difference
////////////////////////////////////////////////////////////////////////

__device__
inline double gaussian (const double width, const double diff)
{
  const double inverse_width = 1.0 / width;
  const double sqrtExponent  = inverse_width * diff;
  const double     exponent  = -sqrtExponent * sqrtExponent;

  return inverse_width * INVERSE_SQRT_PI * expf (exponent);
}


///  Getter for the emissivity (eta) and the opacity (chi)
///    @param[in]     freq_scaled : frequency (in co-moving frame)
///    @param[in]     p           : in dex of the cell
///    @param[in/out] lnotch      : notch indicating location in the set of lines
///    @param[out]    eta         : emissivity
///    @param[out]    chi         : opacity
/////////////////////////////////////////////////////////////////////////////////

__device__
void gpuRayPair :: get_eta_and_chi (const long d, const long f)
{
  const long idf = I(d,f);

  /// Initialize
  eta[idf] = 0.0E+00;
  chi[idf] = 1.0E-26;

  /// Set line emissivity and opacity
  for (long l = 0; l < nlines; l++)
  {
    const long   lnl     = L(nrs[d],l);
    const double diff    = freqs_scaled[idf] - line[l];
    const double profile = freqs_scaled[idf] * gaussian (width[lnl], diff);

    eta[idf] = fma (profile, line_emissivity[lnl], eta[idf]);
    chi[idf] = fma (profile, line_opacity   [lnl], chi[idf]);

//    printf("lnl     = %d\n",  lnl);
//    printf("width   = %le\n",  width[lnl]);
//    printf("diff    = %le\n",  diff);
//    printf("emiciv  = %le\n",  line_emissivity[lnl]);
//    printf("opacit  = %le\n",  line_opacity[lnl]);
//    printf("profile = %le\n", profile);
//
//    printf("eta = %le\n", eta[idf]);
//    printf("chi = %le\n", chi[idf]);
  }

  return;
}


///  Interpolator over frequency space
///    @param[in] Vs : array over frequencies to interpolate
///    @param[in] f  : frequency index at whic hto interpolate
///    @return interpolated value of Vs at index f
//////////////////////////////////////////////////////////////

__device__
double gpuRayPair :: frequency_interpolate (
    const double *Vs,
    const long    f                               )
{
  const long fl = freqs_lower[f];
  const long fu = freqs_upper[f];

  const double t = (freqs_scaled[f]-freqs[fl]) / (freqs[fu]-freqs[fl]);

  return fma(t, Vs[fu], fma(-t, Vs[fl], Vs[fl]));
}


///  Solver for the Feautrier equation along the ray pair
///    @param[in] f : frequency bin index
/////////////////////////////////////////////////////////

__device__
void gpuRayPair :: solve_Feautrier (
    const long f                          )
{
  // printf("Inside solve_Feautrier\n");

  /// SETUP FEAUTRIER RECURSION RELATION
  //////////////////////////////////////


  /// Determine emissivities, opacities and optical depth increments
  TIMER_TIC (t1)
  for (long n = first; n <= last; n++)
  {
    get_eta_and_chi (n, f);
  }
  TIMER_TOC (t1, "get_eta_and_chi")

  TIMER_TIC (t2)
  for (long n = first; n <= last; n++)
  {
    term1[I(n,f)] = eta[I(n,f)] / chi[I(n,f)];
    // printf("term1 = %le\n", term1[I(n,f)]);
  }

  for (long n = first; n <  last; n++)
  {
    dtau[I(n,f)] = 0.5 * (chi[I(n,f)] + chi[I(n+1,f)]) * dZs[n];
    // printf("term1 = %le    dtau = %le\n", term1[I(n,f)], dtau[I(n,f)]);
  }
  TIMER_TOC (t2, "set_term_&_dtau")


  TIMER_TIC (t3)
  /// Set boundary conditions

  const double inverse_dtau0 = 1.0 / dtau[I(first, f)];
  const double inverse_dtaud = 1.0 / dtau[I(last-1,f)];

  C[I(first,f)] = 2.0 * inverse_dtau0 * inverse_dtau0;
  A[I(last, f)] = 2.0 * inverse_dtaud * inverse_dtaud;

  const double B0_min_C0 = fma (2.0, inverse_dtau0, 1.0);
  const double Bd_min_Ad = fma (2.0, inverse_dtaud, 1.0);

  const double B0 = B0_min_C0 + C[I(first,f)];
  const double Bd = Bd_min_Ad + A[I(last, f)];

  const double inverse_B0 = 1.0 / B0;

  const double I_bdy_0 = frequency_interpolate (I_bdy_0_presc, f);
  const double I_bdy_n = frequency_interpolate (I_bdy_n_presc, f);

  // printf ("I_bdy_0 = %le\n", I_bdy_0);
  // printf ("I_bdy_n = %le\n", I_bdy_n);

  Su[I(first,f)] = fma (2.0*I_bdy_0, inverse_dtau0, term1[I(first,f)]);
  Su[I(last, f)] = fma (2.0*I_bdy_n, inverse_dtaud, term1[I(last, f)]);


  /// Set body of Feautrier matrix

  for (long n = first+1; n < last; n++)
  {
    inverse_A[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n-1,f)];
    inverse_C[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n,  f)];

    A[I(n,f)] = 1.0 / inverse_A[I(n,f)];
    C[I(n,f)] = 1.0 / inverse_C[I(n,f)];

    Su[I(n,f)] = term1[I(n,f)];
  }
  TIMER_TOC (t3, "set_up         ")


  /// SOLVE FEAUTRIER RECURSION RELATION
  //////////////////////////////////////


  /// ELIMINATION STEP
  ////////////////////

  TIMER_TIC (t4)

  Su[I(first,f)] = Su[I(first,f)] * inverse_B0;

  // F[0] = (B[0] - C[0]) / C[0];
                   F[I(first,f)] = 0.5 * B0_min_C0 * dtau[I(first,f)] * dtau[I(first,f)];
  inverse_one_plus_F[I(first,f)] = 1.0 / (1.0 + F[I(first,f)]);

  for (long n = first+1; n < last; n++)
  {
                     F[I(n,f)] = (1.0 + A[I(n,f)]*F[I(n-1,f)]*inverse_one_plus_F[I(n-1,f)]) * inverse_C[I(n,f)];
    inverse_one_plus_F[I(n,f)] = 1.0 / (1.0 + F[I(n,f)]);

    Su[I(n,f)] = (Su[I(n,f)] + A[I(n,f)]*Su[I(n-1,f)]) * inverse_one_plus_F[I(n,f)] * inverse_C[I(n,f)];
  }

  const double denominator = 1.0 / fma (Bd, F[I(last-1,f)], Bd_min_Ad);

  Su[I(last,f)] = fma (A[I(last,f)], Su[I(last-1,f)], Su[I(last,f)]) * (1.0 + F[I(last-1,f)]) * denominator;


  /// BACK SUBSTITUTION
  /////////////////////

  if (n_ar < last)
  {
    // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
                    G[I(last,f)] = 0.5 * Bd_min_Ad * dtau[I(last-1,f)] * dtau[I(last-1,f)];
    G_over_one_plus_G[I(last,f)] = G[I(last,f)] / (1.0 + G[I(last,f)]);

    for (long n = last-1; n > n_ar; n--)
    {
      Su[I(n,f)] = fma (Su[I(n+1,f)], inverse_one_plus_F[I(n,f)], Su[I(n,f)]);

                      G[I(n,f)] = (1.0 + C[I(n,f)]*G_over_one_plus_G[I(n+1,f)]) * inverse_A[I(n,f)];
      G_over_one_plus_G[I(n,f)] = G[I(n,f)] / (1.0 + G[I(n,f)]);
    }

    Su[I(n_ar,f)] = fma (Su[I(n_ar+1,f)], inverse_one_plus_F[I(n_ar,f)], Su[I(n_ar,f)]);

    // printf("Su (n_ar) = %le\n", Su[I(n_ar,f)]);

    L_diag[I(n_ar,f)] = inverse_C[I(n_ar,f)] / (F[I(n_ar,f)] + G_over_one_plus_G[I(n_ar+1,f)]);
  }

  else
  {
    L_diag[I(last,f)] = (1.0 + F[I(last-1,f)]) / fma (Bd, F[I(last-1,f)], Bd_min_Ad);
  }

  TIMER_TOC (t4, "solve          ")
  PRINTLINE;


//  if (isnan(Su[I(n_ar,f)]))
//  {
//    for (long n = first; n <  last; n++)
//    {
//      printf("term1 = %le,  dtau = %le,  eta = %le,  chi = %le\n", term1[I(n,f)], dtau[I(n,f)], eta[I(n,f)], chi[I(n,f)]);
//    }
//  }



  return;
}
