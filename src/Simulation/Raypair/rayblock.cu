#include "rayblock.cuh"
#include "Simulation/simulation.hpp"


///  Constructor for RayBlock
///    @param[in] ncells    : number of cells
///    @param[in] nfreqs    : number of frequency bins
///    @param[in] nlines    : number of lines
///    @param[in] nraypairs : number of ray pairs
///    @param[in] depth     : number of points along the ray pairs
//////////////////////////////////////////////////////////////////

CUDA_HOST
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

//    cudaMallocManaged (&I_bdy_0_presc, width_real);
//    cudaMallocManaged (&I_bdy_n_presc, width_real);

    const size_t area_real = 10*depth_max*width_max*sizeof(Real);
    const size_t area_size = 10*depth_max*width_max*sizeof(Size);

//    cudaMallocManaged (&freqs,              area_real);
//    cudaMallocManaged (&freqs_scaled,       area_real);
//    cudaMallocManaged (&freqs_lower,        area_size);
//    cudaMallocManaged (&freqs_upper,        area_size);

    cudaMallocManaged (&term1,              area_real);
    cudaMallocManaged (&term2,              area_real);

    cudaMallocManaged (&eta,                area_real);
    cudaMallocManaged (&chi,                area_real);

    cudaMallocManaged (&A,                  area_real);
    cudaMallocManaged (&a,                  area_real);
    cudaMallocManaged (&C,                  area_real);
    cudaMallocManaged (&c,                  area_real);
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

CUDA_HOST
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

//    cudaFree (I_bdy_0_presc);
//    cudaFree (I_bdy_n_presc);

//    cudaFree (freqs);
//    cudaFree (freqs_scaled);
//    cudaFree (freqs_lower);
//    cudaFree (freqs_upper);

    cudaFree (term1);
    cudaFree (term2);

    cudaFree (eta);
    cudaFree (chi);

    cudaFree (A);
    cudaFree (a);
    cudaFree (C);
    cudaFree (c);
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

CUDA_HOST
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




CUDA_GLOBAL
void feautrierKernel (RayBlock &rayblock)
{
    const Size index  = blockIdx.x * blockDim.x + threadIdx.x;
    const Size stride =  gridDim.x * blockDim.x;

    for (Size w = index; w < rayblock.width; w += stride)
    {
        rayblock.solve_Feautrier (w);
    }
}



CUDA_HOST
void RayBlock :: solve_gpu (
    const ProtoRayBlock &prb,
    const Size           R,
    const Size           r,
          Model         &model    )
{
    /// Setup the ray block with the proto rayblock data
    setup (model, R, r, prb);

    const Size blockSize = gpuBlockSize;
    const Size numBlocks = gpuNumBlocks;

    /// Execute the Feautrier kernel on the GPU
    feautrierKernel <<<numBlocks, blockSize>>> (*this);

    /// Wait for GPU to finish and get possible error
    HANDLE_ERROR (cudaDeviceSynchronize());

    /// Store the result back in the model
    store (model);
}




CUDA_HOST
void RayBlock :: solve_cpu (
    const ProtoRayBlock &prb,
    const Size           R,
    const Size           r,
          Model         &model    )
{
    /// Setup the ray block with the proto rayblock data
    setup (model, R, r, prb);

    /// Execute the Feautrier kernel on the CPU
    for (Size w = 0; w < width; w++)
    {
        solve_Feautrier (w);
    }

    /// Store the result back in the model
    store (model);
}



//CUDA_HOST
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




CUDA_HOST
void RayBlock :: setup (
        const Model           &model,
        const Size             R,
        const Size             r,
        const ProtoRayBlock   &prb   )
{
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

        /// Set n1_min, first and last
        if (n1[rp] < n1_min) n1_min = n1[rp];
        if (fst    < first ) first  = fst;
        if (lst    > last  ) last   = lst;

//        /// Copy boundary intensities
//        cudaMemcpy (&I_bdy_0_presc[V(rp,0)],
//                    model.radiation.I_bdy[RR][bdy_0].data(),
//                    model.radiation.I_bdy[RR][bdy_0].size()*sizeof(double),
//                    cudaMemcpyHostToDevice                                 );
//        cudaMemcpy (&I_bdy_n_presc[V(rp,0)],
//                    model.radiation.I_bdy[RR][bdy_n].data(),
//                    model.radiation.I_bdy[RR][bdy_n].size()*sizeof(double),
//                    cudaMemcpyHostToDevice                                 );
    }


}



CUDA_HOST_DEVICE
inline Real my_fma (const Real a, const Real b, const Real c)
{
    return fma (a, b, c);
}








CUDA_HOST
void RayBlock :: store (Model &model) const
{
    for (Size rp = 0; rp < nraypairs; rp++)
    {
        const double weight_ang = 2.0 * model.geometry.rays.weight(origins[rp], rr);

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

CUDA_HOST_DEVICE
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

CUDA_HOST_DEVICE
inline Real planck (const Real temperature, const Real frequency)
{
    return TWO_HH_OVER_CC_SQUARED * (frequency*frequency*frequency) / expm1 (HH_OVER_KB*frequency/temperature);
}




///  Getter for the emissivity (eta) and the opacity (chi)
///    @param[in]     freq_scaled : frequency (in co-moving frame)
///    @param[in]     p           : in dex of the cell
///    @param[in/out] lnotch      : notch indicating location in the set of lines
///    @param[out]    eta         : emissivity
///    @param[out]    chi         : opacity
/////////////////////////////////////////////////////////////////////////////////

//CUDA_DEVICE
//void RayBlock :: get_eta_and_chi (const Size In, const Size Dn, const Real frequency)
//{
//    const Real frequency_scaled = frequency * shifts[Dn];
//
//    /// Initialize
//    eta[In] = 0.0E+00;
//    chi[In] = 0.0E+00; //1.0E-26;
//
//    /// Set line emissivity and opacity
//    for (Size l = 0; l < nlines; l++)
//    {
//      const Size lnl     = L(nrs[Dn], l);
//      const Real diff    = frequency_scaled - line[l];
//      const Real profile = frequency_scaled * gaussian (line_width[lnl], diff);
//
//      eta[In] = my_fma (profile, line_emissivity[lnl], eta[In]);
//      chi[In] = my_fma (profile, line_opacity   [lnl], chi[In]);
//    }
//}


CUDA_HOST_DEVICE
void RayBlock :: get_eta_and_chi (const Size Dn, const Real frequency, Real &eta, Real &chi)
{
    const Real frequency_scaled = frequency * shifts[Dn];

    /// Initialize
    eta = 0.0E+00;
    chi = 0.0E+00; //1.0E-26;

    /// Set line emissivity and opacity
    for (Size l = 0; l < nlines; l++)
    {
        const Size lnl     = L(nrs[Dn], l);
        const Real diff    = frequency_scaled - line[l];
        const Real profile = frequency_scaled * gaussian (line_width[lnl], diff);

        eta = my_fma (profile, line_emissivity[lnl], eta);
        chi = my_fma (profile, line_opacity   [lnl], chi);
    }
}

//CUDA_DEVICE
//void RayBlock :: get_eta_and_chi (const Size d, const Size w, const Size rp, const Real frequency)
//{
//    const Size idw              = I(d, w);
//    const Real frequency_scaled = frequency * shifts[D(rp, d)];
//
//    /// Initialize
//    eta[idw] = 0.0E+00;
//    chi[idw] = 0.0E+00; //1.0E-26;
//
//    /// Set line emissivity and opacity
//    for (Size l = 0; l < nlines; l++)
//    {
//        const Size lnl     = L(nrs[D(rp, d)], l);
//        const Real diff    = frequency_scaled - line[l];
//        const Real profile = frequency_scaled * gaussian (line_width[lnl], diff);
//
//        eta[idw] = my_fma (profile, line_emissivity[lnl], eta[idw]);
//        chi[idw] = my_fma (profile, line_opacity   [lnl], chi[idw]);
//    }
//}


///  Interpolator over frequency space
///    @param[in] Vs : array over frequencies to interpolate
///    @param[in] f  : frequency index at which to interpolate
///    @return interpolated value of Vs at index f
//////////////////////////////////////////////////////////////

//CUDA_DEVICE
//Real RayBlock :: frequency_interpolate (const Real *Vs, const Size i, const Size w)
//{
//    const Size iw = I(i,w);
//    const Size fl = freqs_lower[iw];
//    const Size fu = freqs_upper[iw];
//    const Real tt = (freqs_scaled[iw]-freqs[fl]) / (freqs[fu]-freqs[fl]);

//    return my_fma(tt, Vs[fu], my_fma(-tt, Vs[fl], Vs[fl]));
//}


///  Solver for the Feautrier equation along the ray pair
///    @param[in] w : width index
/////////////////////////////////////////////////////////

//CUDA_DEVICE
//void RayBlock :: solve_Feautrier (const Size w)
//{
//    TIMER_TIC (t1)
//
//    const Size rp        = w / nfreqs;
//    const Size f         = w % nfreqs;
//    const Real frequency = frequencies[V(origins[rp], f)];
//
//
//    TIMER_TIC (t2)
//
//    const Size If   = I(first,    w);
//    const Size Ifp1 = I(first+1,  w);
//    const Size Df   = D(rp, first);
//
//    get_eta_and_chi (If,   Df,   frequency);
//    get_eta_and_chi (Ifp1, Df+1, frequency);
//
////    Real chi_prev = chi[If];
////    Real chi      = chi[Ifp1];
//
//    term1[If]   = eta[If  ] / chi[If  ];
//    term1[Ifp1] = eta[Ifp1] / chi[Ifp1];
//
//     dtau[If]   = 0.5 * (chi[If] + chi[Ifp1]) * dZs[Df];
//
////    Real dtau_crt = dtau[If];
////    Real dtau_prv = dtau[If];
//
//    /// Set boundary conditions
//    const Real inverse_dtau0 = 1.0 / dtau[If];
//                       C[If] = 2.0 * inverse_dtau0 * inverse_dtau0;
//    const Real B0_min_C0     = my_fma (2.0, inverse_dtau0, 1.0);
//    const Real B0            = B0_min_C0 + C[If];
//
//    const Real inverse_B0 = 1.0 / B0;
//
//    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[Df]);
//    Su[If] = my_fma (2.0*I_bdy_0, inverse_dtau0, term1[If]);
//    Su[If] = Su[If] * inverse_B0;
//
//    // F[0] = (B[0] - C[0]) / C[0];
//    F[If] = 0.5 * B0_min_C0 * dtau[If] * dtau[If];
//    inverse_one_plus_F[If] = 1.0 / (1.0 + F[If]);
//
//
//    TIMER_TIC (t3)
//    /// Set body of Feautrier matrix
//    for (Size n = first+1; n < last; n++)
//    {
//        TIMER_TIC (t5)
//        TIMER_TIC (t11)
//        const Size Inp1 = I(n+1, w);
//        const Size In   = I(n,   w);
//        const Size Inm1 = I(n-1, w);
//        const Size Dn   = D(rp,  n);
//        TIMER_TOC (t11, "indices           ")
//
//        TIMER_TIC (t12)
//        get_eta_and_chi (Inp1, Dn+1, frequency);
//        TIMER_TOC (t12, "get eta and chi   ")
//
//        TIMER_TIC (t13)
//
////        dtau[In  ] = 0.5 * (chi[In] + chi[Inp1]) * dZs[Dn];
//
//        const Real dtau_crt = 0.5 * (chi[In] + chi[Inp1]) * dZs[Dn];
//        const Real  eta_crt = eta[Inp1];
//        const Real  chi_crt = chi[Inp1];
//
//        if (dtau_crt > dtau_max)
//        {
//            const Size n_interpl = dtau_crt / dtau_max + 1;
//            const Real inverse_n = 1.0 / n_interpl;
//
//            const Real ldtau = (dtau_crt - dtau[In]) * inverse_n;
//            const Real  leta = ( eta_crt -  eta[In]) * inverse_n;
//            const Real  lchi = ( chi_crt -  chi[In]) * inverse_n;
//
//            for (Size i = 0; i < n_interpl; i++)
//            {
//                 dtau[In  +i] = (chi[In] + 0.5*i*lchi) * dZs[Dn];
//                term1[Inp1+i] = (eta[In]+i*leta) / (chi[In]+i*lchi);
//            }
//        }
//
////        term1[Inp1] = eta[Inp1] / chi[Inp1];
//
//
//        Su[In] = term1[In];
//
//
//        TIMER_TOC (t13, "term1 and Su      ")
//
//        TIMER_TIC (t14)
//        const Real dtau_av = 0.5 * (dtau[Inm1] + dtau[In]);
//
//        inverse_A[In] = dtau_av * dtau[Inm1];
//                A[In] = 1.0 / inverse_A[In];
////                A[In] = __ddiv_ru (1.0, inverse_A[In]);
//        inverse_C[In] = dtau_av * dtau[In];
//                C[In] = 1.0 / inverse_C[In];
////                C[In] = __ddiv_ru (1.0, inverse_C[In]);
//        TIMER_TOC (t14, "As and Cs         ")
//
//        TIMER_TIC (t15)
//                         F[In] = my_fma (A[In]*F[Inm1], inverse_one_plus_F[Inm1], 1.0) * inverse_C[In];
//        inverse_one_plus_F[In] = 1.0 / (1.0 + F[In]);
//
//        Su[In] = my_fma (A[In], Su[Inm1], Su[In]) * inverse_one_plus_F[In] * inverse_C[In];
//        TIMER_TOC (t15, "final part        ")
//        TIMER_TOC (t5, "elimination loop  ")
//    }
//    TIMER_TOC (t3, "elimination loop  ")
//
//    const Size Il   = I(last,   w);
//    const Size Ilm1 = I(last-1, w);
//    const Size Dl   = D(rp, last);
//
//    const Real inverse_dtaud = 1.0 / dtau[Ilm1];
//                       A[Il] = 2.0 * inverse_dtaud * inverse_dtaud;
//    const Real Bd_min_Ad     = my_fma (2.0, inverse_dtaud, 1.0);
//    const Real Bd            = Bd_min_Ad + A[Il];
//
//    const Real denominator = 1.0 / my_fma (Bd, F[Ilm1], Bd_min_Ad);
//
//    const Real I_bdy_n = planck (T_CMB, frequency*shifts[Dl]);
//    Su[Il] = my_fma (2.0*I_bdy_n, inverse_dtaud, term1[Il]);
//    Su[Il] = my_fma (A[Il], Su[Ilm1], Su[Il]) * (1.0 + F[Ilm1]) * denominator;
//
//    TIMER_TOC (t2, "elimination step  ")
//
//
//    TIMER_TIC (t4)
//
////    if (n1_min < last)
////    {
//        // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
//                        G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
//        G_over_one_plus_G[Il] = G[Il] / (1.0 + G[Il]);
//
//        for (Size n = last-1; n > n1_min; n--)
//        {
//            const Size Inp1 = I(n+1, w);
//            const Size In   = I(n,   w);
//
//                           Su[In] = my_fma (Su[Inp1], inverse_one_plus_F[In], Su[In]);
//
//                            G[In] = my_fma (C[In], G_over_one_plus_G[Inp1], 1.0) * inverse_A[In];
//            G_over_one_plus_G[In] = G[In] / (1.0 + G[In]);
//        }
//
//
//        const Size In1   = I(n1_min,  w);
//        const Size In1p1 = I(n1_min+1,w);
//
//            Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
//        L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);
////    }
////    else
////    {
////        L_diag[Il] = (1.0 + F[Ilm1]) / my_fma (Bd, F[Ilm1], Bd_min_Ad);
////    }
//
//    TIMER_TOC (t4, "back substitution ")
//    TIMER_TOC (t1, "total time solver ")
//    PRINTLINE;
//
//}
//

//CUDA_DEVICE
//void RayBlock :: adapt_optical_depth (const Size w)




//CUDA_DEVICE
//void RayBlock :: solve_Feautrier (const Size w)
//{
//    const Size rp        = w / nfreqs;
//    const Size f         = w % nfreqs;
//    const Real frequency = frequencies[V(origins[rp], f)];
//
//    const Size If   = I(first,    w);
//    const Size Ifp1 = I(first+1,  w);
//    const Size Df   = D(rp, first);
//
////    const Real inverse_dtau_max = 1.0 / 0.05;
//          Size n1_local = n1[rp];
//
//    Real eta_n, eta_1;
//    Real chi_n, chi_1;
//    Real tm1_n, tm1_1;
//
//    get_eta_and_chi (Df,   frequency, eta_n, chi_n);
//    get_eta_and_chi (Df+1, frequency, eta_1, chi_1);
//
//    tm1_n = eta_n / chi_n;
//    tm1_1 = eta_1 / chi_1;
//
//    dtau[If] = 0.5 * (chi_n + chi_1) * dZs[Df];
//
//    /// Set boundary conditions
//    const Real inverse_dtau0 = 1.0 / dtau[If];
//                       C[If] = 2.0 * inverse_dtau0 * inverse_dtau0;
//    const Real B0_min_C0     = my_fma (2.0, inverse_dtau0, 1.0);
//    const Real B0            = B0_min_C0 + C[If];
//
//    const Real inverse_B0 = 1.0 / B0;
//
//    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[Df]);
//    Su[If] = my_fma (2.0*I_bdy_0, inverse_dtau0, tm1_n);
//    Su[If] = Su[If] * inverse_B0;
//
//    // F[0] = (B[0] - C[0]) / C[0];
//    F[If] = 0.5 * B0_min_C0 * dtau[If] * dtau[If];
//    inverse_one_plus_F[If] = 1.0 / (1.0 + F[If]);
//
//
//    Size Inm1  = If;
//    Size In    = Ifp1;
//    Size Inp1  = Ifp1 + width;
//    Size Dn    = Df+1;
//    Size index = first+1;
//
//    /// Set body of Feautrier matrix
//    for (Size n = first+1; n < last; n++)
//    {
//        eta_n = eta_1;
//        chi_n = chi_1;
//
//        get_eta_and_chi (Dn+1, frequency, eta_1, chi_1);
//
//        /// Get the maximum chi value
//        Real chi_max = chi_n;
//        if (chi_max < chi_1) {chi_max = chi_1;}
//
//        /// Get the number of interpolations
//        Size n_interpl = chi_max * dZs[Dn] * inverse_dtau_max + 1;
//        /// Limit the number of interpolations
//        if (n_interpl > 10) {n_interpl = 10;}
//        /// Invert the number of interpolations
//        const Real inverse_n = 1.0 / n_interpl;
//
//        /// Get the index of the result
//        if (n1[rp] == n) {n1_local = index;}
//
//        /// Prepare the interpolation variables
//        const Real ldZs = dZs[Dn]         * inverse_n;
//        const Real leta = (eta_1 - eta_n) * inverse_n;
//        const Real lchi = (chi_1 - chi_n) * inverse_n;
//
//        for (Size i = 1; i <= n_interpl; i++)
//        {
//            eta_1 = eta_n + i*leta;
//            chi_1 = chi_n + i*lchi;
//
//            tm1_n = tm1_1;
//            tm1_1 = eta_1 / chi_1;
//
//            dtau[In] = 0.5 * (chi_n + chi_1) * ldZs;
//              Su[In] = tm1_n;
//
//            const Real dtau_av = 0.5 * (dtau[Inm1] + dtau[In]);
//
//            inverse_A[In] = dtau_av * dtau[Inm1];
//            inverse_C[In] = dtau_av * dtau[In];
//
//            A[In] = 1.0 / inverse_A[In];
//            C[In] = 1.0 / inverse_C[In];
//
//            F[In] = my_fma (A[In]*F[Inm1], inverse_one_plus_F[Inm1], 1.0) * inverse_C[In];
//            inverse_one_plus_F[In] = 1.0 / (1.0 + F[In]);
//
//            Su[In] = my_fma (A[In], Su[Inm1], Su[In]) * inverse_one_plus_F[In] * inverse_C[In];
//
//            Inm1  = In;
//            In    = Inp1;
//            Inp1 += width;
//            index++;
//        }
//
//        Dn++;
//    }
//
//    /// Get the index of the result
//    if (n1[rp] == last) {n1_local = index;}
//
//    const Size Il   = I(index,   w);
//    const Size Ilm1 = I(index-1, w);
//    const Size Dl   = D(rp, last);
//
//    const Real inverse_dtaud = 1.0 / dtau[Ilm1];
//                       A[Il] = 2.0 * inverse_dtaud * inverse_dtaud;
//    const Real Bd_min_Ad     = my_fma (2.0, inverse_dtaud, 1.0);
//    const Real Bd            = Bd_min_Ad + A[Il];
//
//    const Real denominator = 1.0 / my_fma (Bd, F[Ilm1], Bd_min_Ad);
//
//    const Real I_bdy_n = planck (T_CMB, frequency*shifts[Dl]);
//    Su[Il] = my_fma (2.0*I_bdy_n, inverse_dtaud, tm1_1);
//    Su[Il] = my_fma (A[Il], Su[Ilm1], Su[Il]) * (1.0 + F[Ilm1]) * denominator;
//
//
////    if (n1_min < last)
////    {
//    // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
//    G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
//    G_over_one_plus_G[Il] = G[Il] / (1.0 + G[Il]);
//
//    for (Size n = index-1; n > n1_min; n--)
//    {
//        const Size Inp1 = I(n+1, w);
//        const Size In   = I(n,   w);
//
//        Su[In] = my_fma (Su[Inp1], inverse_one_plus_F[In], Su[In]);
//
//                        G[In] = my_fma (C[In], G_over_one_plus_G[Inp1], 1.0) * inverse_A[In];
//        G_over_one_plus_G[In] = G[In] / (1.0 + G[In]);
//    }
//
//
//    const Size In1   = I(n1_min,  w);
//    const Size In1p1 = I(n1_min+1,w);
//
//    Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
//    L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);
////    }
////    else
////    {
////        L_diag[Il] = (1.0 + F[Ilm1]) / my_fma (Bd, F[Ilm1], Bd_min_Ad);
////    }
//
//    Su[I(n1[rp],w)] = Su[I(n1_local,w)];
//
//}



///  Solver for the Feautrier equation along the ray pair
///  method: (Hermitian) 4th-order Feautrier, non-adaptive optical depth increments
///    @param[in] w : width index
/////////////////////////////////////////////////////////

CUDA_HOST_DEVICE
void RayBlock :: solve_Feautrier (const Size w)
{
    const Size rp        = w / nfreqs;
    const Size f         = w % nfreqs;
    const Real frequency = frequencies[V(origins[rp], f)];

    const Size If   = I(first,    w);
    const Size Ifp1 = I(first+1,  w);
    const Size Df   = D(rp, first);

    Size n1_local = n1[rp];

    Real eta_n, eta_1;
    Real chi_n, chi_1;
    Real tm1_n, tm1_1;

    get_eta_and_chi (Df,   frequency, eta_n, chi_n);
    get_eta_and_chi (Df+1, frequency, eta_1, chi_1);

    tm1_n = eta_n / chi_n;
    tm1_1 = eta_1 / chi_1;

    dtau[If] = 0.5 * (chi_n + chi_1) * dZs[Df];

    /// Set boundary conditions
    const Real inverse_dtau0 = 1.0 / dtau[If];

    C[If] = 2.0 * inverse_dtau0 * inverse_dtau0 - ONE_THIRD;

    const Real B0_min_C0 = my_fma (2.0, inverse_dtau0, 1.0);
    const Real B0        = B0_min_C0 + C[If];

    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[Df]);

    Su[If] = 2.0 * I_bdy_0 * inverse_dtau0;
    Su[If] = my_fma (ONE_THIRD,  tm1_1, Su[If]);
    Su[If] = my_fma (TWO_THIRDS, tm1_n, Su[If]);

    /// Start elimination step
    Su[If] /= B0;

    /// Write economically: F[first] = (B[first] - C[first]) / C[first];
                     F[If] = 0.5 * B0_min_C0 * dtau[If] * dtau[If];
    inverse_one_plus_F[If] = 1.0 / (1.0 + F[If]);


    Size Inm1  = If;
    Size In    = Ifp1;
    Size Inp1  = Ifp1 + width;
    Size Dn    = Df+1;
    Size index = first+1;

    /// Set body of Feautrier matrix
    for (Size n = first+1; n < last; n++)
    {
        eta_n = eta_1;
        chi_n = chi_1;

        get_eta_and_chi (Dn+1, frequency, eta_1, chi_1);

        /// Store the previous value of the source function
        Su[In] = tm1_n;

        /// Compute term1 at n+1
        tm1_n = tm1_1;
        tm1_1 = eta_1 / chi_1;

        /// Compute the optical depth increment
        dtau[In] = 0.5 * (chi_n + chi_1) * dZs[Dn];


        const Real dtau_tot = dtau[Inm1] + dtau[In];
        A[In] = 1.0 / (dtau_tot * dtau[Inm1]);
        C[In] = 1.0 / (dtau_tot * dtau[In  ]);

        a[In] = ONE_SIXTH * my_fma (-A[In], dtau[In  ] * dtau[In  ], 1.0);
        c[In] = ONE_SIXTH * my_fma (-C[In], dtau[Inm1] * dtau[Inm1], 1.0);

        A[In] = my_fma (2.0, A[In], -a[In]);
        C[In] = my_fma (2.0, C[In], -c[In]);

        inverse_A[In] = 1.0 / A[In];
        inverse_C[In] = 1.0 / C[In];

        /// Use the previously stored value of the source function
        Su[In] *= a[In];
        Su[In]  = my_fma (1.0 - a[In] - c[In], tm1_n, Su[In]);
        Su[In]  = my_fma (              c[In], tm1_1, Su[In]);


        F[In] = my_fma (A[In]*F[Inm1], inverse_one_plus_F[Inm1], 1.0) * inverse_C[In];
        inverse_one_plus_F[In] = 1.0 / (1.0 + F[In]);

        Su[In] = my_fma (A[In], Su[Inm1], Su[In]) * inverse_one_plus_F[In] * inverse_C[In];

        Inm1  = In;
        In    = Inp1;
        Inp1 += width;
        index++;
        Dn++;
    }


    const Size Il   = I(index,   w);
    const Size Ilm1 = I(index-1, w);
    const Size Dl   = D(rp, last);

    /// Set boundary conditions
    const Real inverse_dtaud = 1.0 / dtau[Ilm1];

    A[Il] = 2.0 * inverse_dtaud * inverse_dtaud - ONE_THIRD;

    const Real Bd_min_Ad = my_fma (2.0, inverse_dtaud, 1.0);
    const Real Bd        = Bd_min_Ad + A[Il];

    const Real denominator = 1.0 / my_fma (Bd, F[Ilm1], Bd_min_Ad);

    const Real I_bdy_n = planck (T_CMB, frequency*shifts[Dl]);

    Su[Il] = 2.0 * I_bdy_n * inverse_dtaud;
    Su[Il] = my_fma (ONE_THIRD,  tm1_n, Su[Il]);
    Su[Il] = my_fma (TWO_THIRDS, tm1_1, Su[Il]);


    Su[Il] = my_fma (A[Il], Su[Ilm1], Su[Il]) * (1.0 + F[Ilm1]) * denominator;


    /// Write economically: G[last] = (B[last] - A[last]) / A[last];
    G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
    G_over_one_plus_G[Il] = G[Il] / (1.0 + G[Il]);


    for (Size n = index-1; n > n1_min; n--)
    {
        const Size Inp1 = I(n+1, w);
        const Size In   = I(n,   w);

        Su[In] = my_fma (Su[Inp1], inverse_one_plus_F[In  ], Su[In]);
         G[In] = my_fma ( C[In  ],  G_over_one_plus_G[Inp1], 1.0   ) * inverse_A[In];

        G_over_one_plus_G[In] = G[In] / (1.0 + G[In]);
    }


    const Size In1   = I(n1_min,   w);
    const Size In1p1 = I(n1_min+1, w);

    Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
    L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);

}


//CUDA_DEVICE
//void RayBlock :: adapt_optical_depth (const Size w)
//{
//
//}


///  Solver for the Feautrier equation along the ray pair
///  method: (Hermitian) 4th-order Feautrier, with adaptive optical depth increments
///    @param[in] w : width index
////////////////////////////////////////////////////////////////////////////////////

//CUDA_DEVICE
////void RayBlock :: solve_4th_order_Feautrier_adaptive (const Size w)
//void RayBlock :: solve_Feautrier (const Size w)
//{
//    const Size rp        = w / nfreqs;
//    const Size f         = w % nfreqs;
//    const Real frequency = frequencies[V(origins[rp], f)];
//
//    const Size If   = I(first,    w);
//    const Size Ifp1 = I(first+1,  w);
//    const Size Df   = D(rp, first);
//
//    Size n1_local = n1[rp];
//
//    Real eta_n, eta_1;
//    Real chi_n, chi_1;
//    Real tm1_n, tm1_1;
//
//    get_eta_and_chi (Df,   frequency, eta_n, chi_n);
//    get_eta_and_chi (Df+1, frequency, eta_1, chi_1);
//
//    tm1_n = eta_n / chi_n;
//    tm1_1 = eta_1 / chi_1;
//
//    dtau[If] = 0.5 * (chi_n + chi_1) * dZs[Df];
//
//    /// Set boundary conditions
//    const Real inverse_dtau0 = 1.0 / dtau[If];
//
//    C[If] = 2.0 * inverse_dtau0 * inverse_dtau0 - ONE_THIRD;
//
//    const Real B0_min_C0 = my_fma (2.0, inverse_dtau0, 1.0);
//    const Real B0        = B0_min_C0 + C[If];
//
//    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[Df]);
//
//    Su[If] = 2.0 * I_bdy_0 * inverse_dtau0;
//    Su[If] = my_fma (ONE_THIRD,  tm1_1, Su[If]);
//    Su[If] = my_fma (TWO_THIRDS, tm1_n, Su[If]);
//
//    /// Start elimination step
//    Su[If] /= B0;
//
//    /// Write economically: F[first] = (B[first] - C[first]) / C[first];
//    F[If] = 0.5 * B0_min_C0 * dtau[If] * dtau[If];
//    inverse_one_plus_F[If] = 1.0 / (1.0 + F[If]);
//
//
//    Size Inm1  = If;
//    Size In    = Ifp1;
//    Size Inp1  = Ifp1 + width;
//    Size Dn    = Df+1;
//    Size index = first+1;
//
//    /// Set body of Feautrier matrix
//    for (Size n = first+1; n < last; n++)
//    {
//        eta_n = eta_1;
//        chi_n = chi_1;
//
//        get_eta_and_chi (Dn+1, frequency, eta_1, chi_1);
//
//        /// Get the maximum chi value
//        Real chi_max;
//        if (chi_n < chi_1) {chi_max = chi_1;}
//        else               {chi_max = chi_n;}
//
//        /// Get the number of interpolations
//        Size n_interpl = chi_max * dZs[Dn] * inverse_dtau_max + 1;
//
//        /// Limit the number of interpolations
//        if (n_interpl > 10) {n_interpl = 10;}
//
//        /// Invert the number of interpolations
//        const Real inverse_n = 1.0 / n_interpl;
//
//        /// Get the index of the result
//        if (n1[rp] == n) {n1_local = index;}
//
//        /// Prepare the interpolation variables
//        const Real ldZs = dZs[Dn]         * inverse_n;
//        const Real leta = (eta_1 - eta_n) * inverse_n;
//        const Real lchi = (chi_1 - chi_n) * inverse_n;
//
//        printf("n_interpl = %ld\n", n_interpl);
//
//        for (Size i = 1; i <= n_interpl; i++)
//        {
//            /// Linearly interpolate the emissivity and the opacity
//            eta_1 = my_fma (i, leta, eta_n);
//            chi_1 = my_fma (i, lchi, chi_n);
//
//            /// Store the previous value of the source function
//            Su[In] = tm1_n;
//
//            /// Compute term1 at n+1
//            tm1_n = tm1_1;
//            tm1_1 = eta_1 / chi_1;
//
//            /// Compute the optical depth increment
//            dtau[In] = 0.5 * (chi_n + chi_1) * ldZs;
//
//            const Real dtau_tot = dtau[Inm1] + dtau[In];
//
//            const Real AA = 1.0 / (dtau_tot * dtau[Inm1]);
//            const Real CC = 1.0 / (dtau_tot * dtau[In  ]);
//
//            a[In] = ONE_SIXTH * my_fma (-AA, dtau[In  ] * dtau[In  ], 1.0);
//            c[In] = ONE_SIXTH * my_fma (-CC, dtau[Inm1] * dtau[Inm1], 1.0);
//
//            A[In] = my_fma (2.0, AA, -a[In]);
//            C[In] = my_fma (2.0, CC, -c[In]);
//
//            inverse_A[In] = 1.0 / A[In];
//            inverse_C[In] = 1.0 / C[In];
//
//            /// Use the previously stored value of the source function
//            Su[In] *= a[In];
//            Su[In]  = my_fma (1.0 - a[In] - c[In], tm1_n, Su[In]);
//            Su[In]  = my_fma (              c[In], tm1_1, Su[In]);
//
//            F[In] = my_fma (A[In]*F[Inm1], inverse_one_plus_F[Inm1], 1.0) * inverse_C[In];
//            inverse_one_plus_F[In] = 1.0 / (1.0 + F[In]);
//
//            Su[In] = my_fma (A[In], Su[Inm1], Su[In]) * inverse_one_plus_F[In] * inverse_C[In];
//
//            Inm1  = In;
//            In    = Inp1;
//            Inp1 += width;
//            index++;
//        }
//
//        Dn++;
//    }
//
//
//    /// Get the index of the result
//    if (n1[rp] == last) {n1_local = index;}
//
//
//    const Size Il   = I(index,   w);
//    const Size Ilm1 = I(index-1, w);
//    const Size Dl   = D(rp, last);
//
//    /// Set boundary conditions
//    const Real inverse_dtaud = 1.0 / dtau[Ilm1];
//
//    A[Il] = 2.0 * inverse_dtaud * inverse_dtaud - ONE_THIRD;
//
//    const Real Bd_min_Ad = my_fma (2.0, inverse_dtaud, 1.0);
//    const Real Bd        = Bd_min_Ad + A[Il];
//
//    const Real denominator = 1.0 / my_fma (Bd, F[Ilm1], Bd_min_Ad);
//
//    const Real I_bdy_n = planck (T_CMB, frequency*shifts[Dl]);
//
//    Su[Il] = 2.0 * I_bdy_n * inverse_dtaud;
//    Su[Il] = my_fma (ONE_THIRD,  tm1_n, Su[Il]);
//    Su[Il] = my_fma (TWO_THIRDS, tm1_1, Su[Il]);
//
//
//    Su[Il] = my_fma (A[Il], Su[Ilm1], Su[Il]) * (1.0 + F[Ilm1]) * denominator;
//
//
//    /// Write economically: G[last] = (B[last] - A[last]) / A[last];
//    G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
//    G_over_one_plus_G[Il] = G[Il] / (1.0 + G[Il]);
//
//
//    for (Size n = index-1; n > n1_min; n--)
//    {
//        const Size Inp1 = I(n+1, w);
//        const Size In   = I(n,   w);
//
//        Su[In] = my_fma (Su[Inp1], inverse_one_plus_F[In  ], Su[In]);
//         G[In] = my_fma ( C[In  ],  G_over_one_plus_G[Inp1],    1.0) * inverse_A[In];
//
//        G_over_one_plus_G[In] = G[In] / (1.0 + G[In]);
//    }
//
//
//    const Size In1   = I(n1_min,   w);
//    const Size In1p1 = I(n1_min+1, w);
//
//    Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
//    L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);
//
//
//    // TODO: Avoid this remapping.
//
//    /// Map the result to the expected place
//    Su[I(n1[rp],w)] = Su[I(n1_local,w)];
//
//}
