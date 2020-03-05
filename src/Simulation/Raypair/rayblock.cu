#include "rayblock.cuh"
#include "Simulation/simulation.hpp"


///  Constructor for RayBlock
///    @param[in] ncells    : number of cells
///    @param[in] nfreqs    : number of frequency bins
///    @param[in] nlines    : number of lines
///    @param[in] nraypairs : number of ray pairs
///    @param[in] depth     : number of points along the ray pairs
//////////////////////////////////////////////////////////////////

__host__
RayBlock ::  RayBlock (
    const Size ncells,
    const Size nfreqs,
    const Size nlines,
    const Size nraypairs,
    const Size depth     )
    : ncells        (ncells)
    , nfreqs        (nfreqs)
    , nlines        (nlines)
    , nraypairs_max (nraypairs)
    , nraypairs     (nraypairs)
    , depth_max     (depth)
    , width_max     (nraypairs * nfreqs)
    , width         (nraypairs * nfreqs)
{
    const size_t nraypairs_size = nraypairs_max*sizeof(Size);
    const size_t nraypairs_real = nraypairs_max*sizeof(Real);

    cudaMallocManaged (&n1,     nraypairs_size);
    cudaMallocManaged (&n2,     nraypairs_size);

    cudaMallocManaged (&origins, nraypairs_size);
    cudaMallocManaged (&reverse, nraypairs_real);

    cudaMallocManaged (&nrs,    depth_max*nraypairs_max*sizeof(Size));
    cudaMallocManaged (&shifts, depth_max*nraypairs_max*sizeof(Real));
    cudaMallocManaged (&dZs,    depth_max*nraypairs_max*sizeof(Real));

    cudaMallocManaged (&line,                   nlines*sizeof(double));
    cudaMallocManaged (&line_emissivity, ncells*nlines*sizeof(double));
    cudaMallocManaged (&line_opacity,    ncells*nlines*sizeof(double));
    cudaMallocManaged (&line_width,      ncells*nlines*sizeof(double));

    cudaMallocManaged (&frequencies,     ncells*nfreqs*sizeof(double));

    const size_t width_real = width_max*sizeof(Real);

    cudaMallocManaged (&I_bdy_0_presc, width_real);
    cudaMallocManaged (&I_bdy_n_presc, width_real);

    const size_t area_real = depth_max*width_max*sizeof(Real);
    const size_t area_size = depth_max*width_max*sizeof(Size);

//    cudaMallocManaged (&freqs,              area_real);
//    cudaMallocManaged (&freqs_scaled,       area_real);
//    cudaMallocManaged (&freqs_lower,        area_size);
//    cudaMallocManaged (&freqs_upper,        area_size);

    cudaMallocManaged (&term1,              area_real);
    cudaMallocManaged (&term2,              area_real);

    cudaMallocManaged (&eta,                area_real);
    cudaMallocManaged (&chi,                area_real);

    cudaMallocManaged (&A,                  area_real);
    cudaMallocManaged (&C,                  area_real);
    cudaMallocManaged (&F,                  area_real);
    cudaMallocManaged (&G,                  area_real);

    cudaMallocManaged (&inverse_A,          area_real);
    cudaMallocManaged (&inverse_C,          area_real);
    cudaMallocManaged (&inverse_one_plus_F, area_real);
    cudaMallocManaged (&inverse_one_plus_G, area_real);
    cudaMallocManaged (& G_over_one_plus_G, area_real);

    cudaMallocManaged (&Su,                 area_real);
    cudaMallocManaged (&Sv,                 area_real);
    cudaMallocManaged (&dtau,               area_real);

    cudaMallocManaged (&L_diag,             area_real);
}


///  Destructor for gpuRayPair
//////////////////////////////

__host__
RayBlock :: ~RayBlock ()
{
    cudaFree (n1);
    cudaFree (n2);

    cudaFree (origins);
    cudaFree (reverse);

    cudaFree (nrs);
    cudaFree (shifts);
    cudaFree (dZs);

    cudaFree (line);
    cudaFree (line_emissivity);
    cudaFree (line_opacity);
    cudaFree (line_width);

    cudaFree (frequencies);

    cudaFree (I_bdy_0_presc);
    cudaFree (I_bdy_n_presc);

//    cudaFree (freqs);
//    cudaFree (freqs_scaled);
//    cudaFree (freqs_lower);
//    cudaFree (freqs_upper);

    cudaFree (term1);
    cudaFree (term2);

    cudaFree (eta);
    cudaFree (chi);

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
}




///  Copies the model data into the gpuRayPair data structure
///    @param[in] model : model from which to copy
/////////////////////////////////////////////////////////////

void RayBlock :: copy_model_data (const Model &model)
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


    for (Size p = 0; p < ncells; p++)
    {
        for (Size l = 0; l < nlines; l++)
        {
            const Size         f = model.lines.line_index[l];
            const Size     lspec = model.radiation.frequencies.corresponding_l_for_spec[f];
            const Real invr_mass = model.lines.lineProducingSpecies[lspec].linedata.inverse_mass;

            line_width[L(p,l)] = model.thermodynamics.profile_width (invr_mass, p, model.lines.line[l]);
        }

        for (Size f = 0; f < nfreqs; f++)
        {
            frequencies[V(p,f)] = model.radiation.frequencies.nu[p][f];
        }
    }

}




//__host__
//void RayBlock :: setFrequencies (
//        const Double1 &frequencies,
//        const Real     scale,
//        const Size     index,
//        const Size     rp          )
//{
//    timer1.start();

//    for (Size rp = 0; rp < nraypairs; rp++)
//    {
//        Size f1 = 0;

//        for (Size f2 = 0; f2 < nfreqs; f2++)
//        {
//            const Size if2 = I(index, V(rp, f2));

//            freqs       [if2] = frequencies[f2];
//            freqs_scaled[if2] = frequencies[f2] * scale;

////            while (   (frequencies[f1] < freqs_scaled[if2])
////                   && (f1              < nfreqs-1         ) )
////            {
////                f1++;
////            }
////
////            if (f1 > 0)
////            {
////                freqs_lower[if2] = f1-1;
////                freqs_upper[if2] = f1;
////            }
////            else
////            {
////                freqs_lower[if2] = 0;
////                freqs_upper[if2] = 1;
////            }
//        }
//    }

//    timer1.stop();
//}




__host__
void RayBlock :: setup (
        const Model           &model,
        const Size             R,
        const Size             r,
        const ProtoRayBlock   &prb   )
{
    timer0.start();

    /// Set the ray direction indices
    RR = R;
    rr = r;

    /// Initialize n1_min, first and last
    n1_min = depth_max;
    first  = depth_max;
    last   = 0;

    for (Size rp = 0; rp < nraypairs; rp++)
    {
        /// Set the origin
        Size o = origins[rp] = prb.origins[rp];

        /// For interchanging ray data
        RayData raydata1;
        RayData raydata2;

        /// Ensure that ray 1 is longer than ray 2
        if (prb.rays_ar[rp].size() > prb.rays_rr[rp].size())
        {
            reverse[rp] =            +1.0;
            raydata1    = prb.rays_ar[rp];
            raydata2    = prb.rays_rr[rp];
        }
        else
        {
            reverse[rp] =            -1.0;
            raydata1    = prb.rays_rr[rp];
            raydata2    = prb.rays_ar[rp];
        }

        /// Extract ray and antipodal ray lengths
        n1[rp] = raydata1.size();
        n2[rp] = raydata2.size();

        /// Initialize index
        Size index = n1[rp];

        /// Temporary variables for first and last
        Size fst = 0;
        Size lst = n1[rp];

        /// Set origin
//        setFrequencies (model.radiation.frequencies.nu[o], 1.0, index, rp);
        nrs   [D(rp,index)] = o;
        shifts[D(rp,index)] = 1.0;

        /// Temporary boundary numbers
        Size bdy_0 = model.geometry.boundary.cell2boundary_nr[o];
        Size bdy_n = model.geometry.boundary.cell2boundary_nr[o];

        timer2.start();
        /// Set ray 1
        if (n1[rp] > 0)
        {
            index = n1[rp]-1;

            for (const ProjectedCellData &data : raydata1)
            {
//                setFrequencies (model.radiation.frequencies.nu[o], data.shift, index, rp);
                nrs   [D(rp,index)] = data.cellNr;
                shifts[D(rp,index)] = data.shift;
                dZs   [D(rp,index)] = data.dZ;
                index--;
            }

            bdy_0 = model.geometry.boundary.cell2boundary_nr[raydata1.back().cellNr];
            fst   = index+1;
        }

        /// Set ray 2
        if (n2[rp] > 0)
        {
            index = n1[rp]+1;

            for (const ProjectedCellData &data : raydata2)
            {
//                setFrequencies (model.radiation.frequencies.nu[o], data.shift, index, rp);
                nrs   [D(rp,index  )] = data.cellNr;
                shifts[D(rp,index  )] = data.shift;
                dZs   [D(rp,index-1)] = data.dZ;
                index++;
            }

            bdy_n = model.geometry.boundary.cell2boundary_nr[raydata2.back().cellNr];
            lst   = index-1;
        }
        timer2.stop();

        /// Set n1_min, first and last
        if (n1[rp] < n1_min) n1_min = n1[rp];
        if (fst    < first ) first  = fst;
        if (lst    > last  ) last   = lst;

//        timer3.start();
//        /// Copy boundary intensities
//        cudaMemcpy (&I_bdy_0_presc[V(rp,0)],
//                    model.radiation.I_bdy[RR][bdy_0].data(),
//                    model.radiation.I_bdy[RR][bdy_0].size()*sizeof(double),
//                    cudaMemcpyHostToDevice                                 );
//        cudaMemcpy (&I_bdy_n_presc[V(rp,0)],
//                    model.radiation.I_bdy[RR][bdy_n].data(),
//                    model.radiation.I_bdy[RR][bdy_n].size()*sizeof(double),
//                    cudaMemcpyHostToDevice                                 );
//        timer3.stop();
    }

    timer0.stop();

}




__global__
void feautrierKernel (RayBlock &rayblock)
{
    const Size index  = blockIdx.x * blockDim.x + threadIdx.x;
    const Size stride =  gridDim.x * blockDim.x;

    for (Size w = index; w < rayblock.width; w += stride)
    {
        rayblock.solve_Feautrier (w);
    }
}




__host__
void RayBlock :: solve ()
{
    const Size blockSize = gpuBlockSize;
    const Size numBlocks = (width + blockSize-1) / blockSize;

    feautrierKernel <<<numBlocks, blockSize>>> (*this);

    // Wait for GPU to finish and get possible error
    HANDLE_ERROR (cudaDeviceSynchronize());
}




__host__
void RayBlock :: store (Model &model) const
{
    const double weight_ang = 2.0 * model.geometry.rays.weights[rr];

    for (Size rp = 0; rp < nraypairs; rp++)
    {
        const Size i0 = model.radiation.index(origins[rp], 0);
        const Size j0 = I(n1[rp], V(rp, 0));

        for (Size f = 0; f < nfreqs; f++)
        {
            model.radiation.J[i0+f] += weight_ang * Su[j0+f];
        }

        if (model.parameters.use_scattering())
        {
            for (Size f = 0; f < nfreqs; f++)
            {
                model.radiation.u[RR][i0+f] = Su[j0+f];
                model.radiation.v[RR][i0+f] = Sv[j0+f] * reverse[rp];
            }
        }
    }
}




///  Gaussian line profile function
///    @param[in] width     : profile width
///    @param[in] freq_diff : frequency difference with line centre
///    @return profile function evaluated with this frequency difference
////////////////////////////////////////////////////////////////////////

__device__
inline Real gaussian (const Real width, const Real diff)
{
    const Real inverse_width = 1.0 / width;
    const Real sqrtExponent  = inverse_width * diff;
    const Real     exponent  = -sqrtExponent * sqrtExponent;

    return inverse_width * INVERSE_SQRT_PI * expf (exponent);
}




///  Planck function
///    @param[in] temperature : temperature of the corresponding black body
///    @param[in] frequency   : frequency at which to evaluate the function
///    @return Planck function evaluated at this frequency
///////////////////////////////////////////////////////////////////////////

__device__
inline Real planck (const Real temperature, const Real frequency)
{
    return TWO_HH_OVER_CC_SQUARED * (frequency*frequency*frequency) / expm1(HH_OVER_KB*frequency/temperature);
}




///  Getter for the emissivity (eta) and the opacity (chi)
///    @param[in]     freq_scaled : frequency (in co-moving frame)
///    @param[in]     p           : in dex of the cell
///    @param[in/out] lnotch      : notch indicating location in the set of lines
///    @param[out]    eta         : emissivity
///    @param[out]    chi         : opacity
/////////////////////////////////////////////////////////////////////////////////

__device__
void RayBlock :: get_eta_and_chi (const Size d, const Size w, const Size rp, const Real frequency)
{
    const Size idw              = I(d, w);
    const Real frequency_scaled = frequency * shifts[D(rp, d)];

    /// Initialize
    eta[idw] = 0.0E+00;
    chi[idw] = 0.0E+00; //1.0E-26;

    /// Set line emissivity and opacity
    for (Size l = 0; l < nlines; l++)
    {
      const Size lnl     = L(nrs[D(rp, d)], l);
      const Real diff    = frequency_scaled - line[l];
      const Real profile = frequency_scaled * gaussian (line_width[lnl], diff);

      eta[idw] = fma (profile, line_emissivity[lnl], eta[idw]);
      chi[idw] = fma (profile, line_opacity   [lnl], chi[idw]);
    }
}


///  Interpolator over frequency space
///    @param[in] Vs : array over frequencies to interpolate
///    @param[in] f  : frequency index at which to interpolate
///    @return interpolated value of Vs at index f
//////////////////////////////////////////////////////////////

//__device__
//Real RayBlock :: frequency_interpolate (const Real *Vs, const Size i, const Size w)
//{
//    const Size iw = I(i,w);
//    const Size fl = freqs_lower[iw];
//    const Size fu = freqs_upper[iw];
//    const Real tt = (freqs_scaled[iw]-freqs[fl]) / (freqs[fu]-freqs[fl]);

//    return fma(tt, Vs[fu], fma(-tt, Vs[fl], Vs[fl]));
//}


///  Solver for the Feautrier equation along the ray pair
///    @param[in] w : width index
/////////////////////////////////////////////////////////

__device__
void RayBlock :: solve_Feautrier (const Size w)
{

    // printf("Inside solve_Feautrier\n");
    const Size rp        = w / nfreqs;
    const Size f         = w % nfreqs;
    const Real frequency = frequencies[V(origins[rp], f)];

    /// SETUP FEAUTRIER RECURSION RELATION
    //////////////////////////////////////


    TIMER_TIC (t1)
    /// Determine emissivities, opacities and optical depth increments
    for (Size n = first; n <= last; n++)
    {
        get_eta_and_chi (n, w, rp, frequency);
    }
    TIMER_TOC (t1, "get_eta_and_chi")


    TIMER_TIC (t2)
    for (Size n = first; n <= last; n++)
    {
        term1[I(n,w)] = eta[I(n,w)] / chi[I(n,w)];
//         printf("term1 = %le\n", term1[I(n,w)]);
    }

    for (Size n = first; n <  last; n++)
    {
        dtau[I(n,w)] = 0.5 * (chi[I(n,w)] + chi[I(n+1,w)]) * dZs[D(rp,n)];
//        printf("%d : term1 = %le, dtau = %le, shift = %le, freq = %le, origin =%d, f=%ld\n", n, term1[I(n,w)], dtau[I(n,w)], shifts[D(rp, n)], frequency, origins[rp], w);
    }
    TIMER_TOC (t2, "set_term_&_dtau")


    TIMER_TIC (t3)

    /// Set boundary conditions
    const Real inverse_dtau0 = 1.0 / dtau[I(first, w)];
    const Real inverse_dtaud = 1.0 / dtau[I(last-1,w)];

    C[I(first,w)] = 2.0 * inverse_dtau0 * inverse_dtau0;
    A[I(last, w)] = 2.0 * inverse_dtaud * inverse_dtaud;

    const Real B0_min_C0 = fma (2.0, inverse_dtau0, 1.0);
    const Real Bd_min_Ad = fma (2.0, inverse_dtaud, 1.0);

    const Real B0 = B0_min_C0 + C[I(first,w)];
    const Real Bd = Bd_min_Ad + A[I(last, w)];

    const Real inverse_B0 = 1.0 / B0;

//    const Real I_bdy_0 = frequency_interpolate (I_bdy_0_presc, first, w);
//    const Real I_bdy_n = frequency_interpolate (I_bdy_n_presc, last,  w);

    TIMER_TIC (t13)
    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[D(rp, first)]);
    const Real I_bdy_n = planck (T_CMB, frequency*shifts[D(rp, last )]);
    TIMER_TOC (t13, "planck         ")

    // printf ("I_bdy_0 = %le\n", I_bdy_0);
    // printf ("I_bdy_n = %le\n", I_bdy_n);

    Su[I(first,w)] = fma (2.0*I_bdy_0, inverse_dtau0, term1[I(first,w)]);
    Su[I(last, w)] = fma (2.0*I_bdy_n, inverse_dtaud, term1[I(last, w)]);


    /// Set body of Feautrier matrix
    for (Size n = first+1; n < last; n++)
    {
        inverse_A[I(n,w)] = 0.5 * (dtau[I(n-1,w)] + dtau[I(n,w)]) * dtau[I(n-1,w)];
        inverse_C[I(n,w)] = 0.5 * (dtau[I(n-1,w)] + dtau[I(n,w)]) * dtau[I(n,  w)];

        A[I(n,w)] = 1.0 / inverse_A[I(n,w)];
        C[I(n,w)] = 1.0 / inverse_C[I(n,w)];

        Su[I(n,w)] = term1[I(n,w)];
    }

    TIMER_TOC (t3, "set_up         ")


    /// SOLVE FEAUTRIER RECURSION RELATION
    //////////////////////////////////////


    /// ELIMINATION STEP
    ////////////////////

    TIMER_TIC (t4)

    Su[I(first,w)] = Su[I(first,w)] * inverse_B0;

    // F[0] = (B[0] - C[0]) / C[0];
                     F[I(first,w)] = 0.5 * B0_min_C0 * dtau[I(first,w)] * dtau[I(first,w)];
    inverse_one_plus_F[I(first,w)] = 1.0 / (1.0 + F[I(first,w)]);

    for (Size n = first+1; n < last; n++)
    {
                         F[I(n,w)] = (1.0 + A[I(n,w)]*F[I(n-1,w)]*inverse_one_plus_F[I(n-1,w)]) * inverse_C[I(n,w)];
        inverse_one_plus_F[I(n,w)] = 1.0 / (1.0 + F[I(n,w)]);

        Su[I(n,w)] = (Su[I(n,w)] + A[I(n,w)]*Su[I(n-1,w)]) * inverse_one_plus_F[I(n,w)] * inverse_C[I(n,w)];
    }

    const Real denominator = 1.0 / fma (Bd, F[I(last-1,w)], Bd_min_Ad);

    Su[I(last,w)] = fma (A[I(last,w)], Su[I(last-1,w)], Su[I(last,w)]) * (1.0 + F[I(last-1,w)]) * denominator;


    /// BACK SUBSTITUTION
    /////////////////////

    if (n1_min < last)
    {
        // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
                        G[I(last,w)] = 0.5 * Bd_min_Ad * dtau[I(last-1,w)] * dtau[I(last-1,w)];
        G_over_one_plus_G[I(last,w)] = G[I(last,w)] / (1.0 + G[I(last,w)]);

        for (Size n = last-1; n > n1_min; n--)
        {
            Su[I(n,w)] = fma (Su[I(n+1,w)], inverse_one_plus_F[I(n,w)], Su[I(n,w)]);

                            G[I(n,w)] = (1.0 + C[I(n,w)]*G_over_one_plus_G[I(n+1,w)]) * inverse_A[I(n,w)];
            G_over_one_plus_G[I(n,w)] = G[I(n,w)] / (1.0 + G[I(n,w)]);
        }

        Su[I(n1_min,w)] = fma (Su[I(n1_min+1,w)], inverse_one_plus_F[I(n1_min,w)], Su[I(n1_min,w)]);

        // printf("Su (n_ar) = %le\n", Su[I(n_ar,f)]);

        L_diag[I(n1_min,w)] = inverse_C[I(n1_min,w)] / (F[I(n1_min,w)] + G_over_one_plus_G[I(n1_min+1,w)]);
    }
    else
    {
        L_diag[I(last,w)] = (1.0 + F[I(last-1,w)]) / fma (Bd, F[I(last-1,w)], Bd_min_Ad);
    }

    TIMER_TOC (t4, "solve          ")
    PRINTLINE;


//    if (isnan(Su[I(n1_min,w)]))
//    {
//      for (long n = first; n <  last; n++)
//      {
//        printf("term1 = %le,  dtau = %le,  eta = %le,  chi = %le\n", term1[I(n,w)], dtau[I(n,w)], eta[I(n,w)], chi[I(n,w)]);
//      }
//    }
}
