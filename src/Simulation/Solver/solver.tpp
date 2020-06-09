inline void Solver :: setup (
        const Model        &model,
        const Size          R,
        const Size          r,
        const ProtoBlock   &prb   )
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




static inline Real my_fma (const Real a, const Real b, const Real c)
{
    return fma (a, b, c);
}


///  Store the result of the solver in the model
///    @param[in/out] model : model object under consideration
//////////////////////////////////////////////////////////////

inline void Solver :: store (Model &model) const
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
//                model.radiation.v[RR][i0+f] = Sv[j0+f] * reverse[rp];
            }
        }
    }
}




///  Gaussian line profile function
///    @param[in] width     : profile width
///    @param[in] freq_diff : frequency difference with line centre
///    @return profile function evaluated with this frequency difference
////////////////////////////////////////////////////////////////////////

static inline Real my_gaussian (const Real width, const Real diff)
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

static inline Real my_planck (const Real temperature, const Real frequency)
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

HOST_DEVICE
inline void Solver :: get_eta_and_chi (const Size Dn, const Real frequency, Real &eta, Real &chi)
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
        const Real profile = frequency_scaled * my_gaussian (line_width[lnl], diff);

        eta = my_fma (profile, line_emissivity[lnl], eta);
        chi = my_fma (profile, line_opacity   [lnl], chi);
    }
}




///  Solver for the Feautrier equation along the ray pair
///  method: (Hermitian) 4th-order Feautrier, non-adaptive optical depth increments
///    @param[in] w : width index
/////////////////////////////////////////////////////////

HOST_DEVICE
inline void Solver :: solve_Feautrier (const Size w)
{
    const Size rp        = w / nfreqs;
    const Size f         = w % nfreqs;
    const Real frequency = frequencies[V(origins[rp], f)];

    const Size If   = I(first,   w);
    const Size Ifp1 = I(first+1, w);
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

    const Real I_bdy_0 = my_planck (T_CMB, frequency*shifts[Df]);

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

    const Real I_bdy_n = my_planck (T_CMB, frequency*shifts[Dl]);

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
