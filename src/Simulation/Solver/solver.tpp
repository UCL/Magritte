
#include <math.h>

template <typename Real>
inline void Solver<Real> :: setup (
        const Model        &model,
        const Size          R,
        const Size          r,
        const ProtoBlock   &prb    )
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

        /// Set n_tot
        n_tot[rp] = n1[rp] + n2[rp] + 1;

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
    }

}


template <typename Real>
HOST_DEVICE
inline Real Solver<Real> :: my_fma (const Real a, const Real b, const Real c) const
{
//    return fma (a, b, c);
    return a*b + c;
}




//template <typename Real>
//HOST_DEVICE
//inline Real Solver<Real> :: expf (const Real x) const
//{
//    const int n = 11;
//    Real result = one;
//
//    for (int i = n; i > 1; i--)
//    {
//        result = one - x*result*inverse_index[i];
//    }
//
//    return 1.0 / (one - x*result);
//}
//
//template <typename Real>
//HOST_DEVICE
//inline Real Solver<Real> :: expm1 (const Real x) const
//{
//    const int n = 11;
//    Real result = one;
//
//    for (int i = n; i > 1; i--)
//    {
//        result = one + x*result*inverse_index[i];
//    }
//
//    return x*result;
//}




template <typename Real>
inline void Solver<Real> :: add_L_diag (
    const Thermodynamics &thermodyn,
    const double          invr_mass,
    const double          freq_line,
    const double          constante,
    const Size            rp,
    const Size            f,
    const Size            k,
          Lambda         &lambda       ) const
{
    const Size In = I(n1[rp], V(rp, f));
    const Size Dn = D(rp    , n1[rp] );

    const Real fsc = frequencies[V(nrs[Dn], f)] * shifts[Dn];
    const Real phi = thermodyn.profile (invr_mass, nrs[Dn], freq_line, fsc);

    const Real L = constante * fsc * phi * L_diag[In] / chi[In];

//    if(isnan(L)) {printf("!!! L d is nan %le, %le\n", L_diag[In], chi[In]);}

//    printf("eta / eta = %le", HH_OVER_FOUR_PI * constante * fsc * phi / eta[In]);

//    const Real L = eta[In] * L_diag[In] / (HH_OVER_FOUR_PI * chi[In]);

//    printf("L = %le\n", L);


    lambda.add_element(origins[rp], k, nrs[Dn], L);
}




template <typename Real>
inline void Solver<Real> :: add_L_lower (
    const Thermodynamics &thermodyn,
    const double          invr_mass,
    const double          freq_line,
    const double          constante,
    const Size            rp,
    const Size            f,
    const Size            k,
    const Size            m,
          Lambda         &lambda        ) //const
{
    const Size In = I(n1[rp]-m-1, V(rp, f));
    const Size Dn = D(rp      , n1[rp]-m-1);

    const Real fsc = frequencies[V(nrs[Dn], f)] * shifts[Dn];
    const Real phi = thermodyn.profile (invr_mass, nrs[Dn], freq_line, fsc);

    const Real L = constante * fsc * phi * L_lower[M(m,In)] / chi[In];

//    if(isnan(L))
//    {
//        for (Size w = 0; w < width; w++) {check_L(w);}
//
//        printf("!!! L l is nan, M(%ld, %ld) = %ld, n_off_diag = %ld, last = %ld, n_tot = %ld\n", m, In, M(m,In), n_off_diag, last, n_tot[rp]);
//    }

//    printf("eta / eta = %le", HH_OVER_FOUR_PI * constante * fsc * phi / eta[In]);

//    const Real L = eta[In] * L_lower[M(m,In)] / (HH_OVER_FOUR_PI * chi[In]);

    lambda.add_element(origins[rp], k, nrs[Dn], L);
}




template <typename Real>
inline void Solver<Real> :: add_L_upper (
    const Thermodynamics &thermodyn,
    const double          invr_mass,
    const double          freq_line,
    const double          constante,
    const Size            rp,
    const Size            f,
    const Size            k,
    const Size            m,
          Lambda         &lambda        ) //const
{
    const Size In = I(n1[rp]+m+1, V(rp, f));
    const Size Dn = D(rp      , n1[rp]+m+1);

    const Real fsc = frequencies[V(nrs[Dn], f)] * shifts[Dn];
    const Real phi = thermodyn.profile (invr_mass, nrs[Dn], freq_line, fsc);

    const Real L = constante * fsc * phi * L_upper[M(m,In)] / chi[In];

//    if(isnan(L)) {
//
//        for (Size w = 0; w < width; w++) {check_L(w);}
//
//        printf("!!! L u is nan, M(%ld, %ld) = %ld, n_off_diag = %ld, last = %ld, n_tot = %ld\n", m, In, M(m,In), n_off_diag, last, n_tot[rp]);
//    }

//    printf("eta / eta = %le", HH_OVER_FOUR_PI * constante * fsc * phi / eta[In]);

//    const Real L = eta[In] * L_upper[M(m,In)] / (HH_OVER_FOUR_PI * chi[In]);

    lambda.add_element(origins[rp], k, nrs[Dn], L);
}


template <typename Real>
inline void Solver<Real> :: update_Lambda (Model &model) //const
{
    const Frequencies    &freqs     = model.radiation.frequencies;
    const Thermodynamics &thermodyn = model.thermodynamics;

    for (Size rp = 0; rp < nraypairs; rp++)
    {
        const double w_ang = 2.0 * model.geometry.rays.weight(origins[rp], rr);

        for (Size f = 0; f < nfreqs; f++) if (freqs.appears_in_line_integral[f])
        {
            const Size l = freqs.corresponding_l_for_spec[f];   // index of species
            const Size k = freqs.corresponding_k_for_tran[f];   // index of transition
            const Size z = freqs.corresponding_z_for_line[f];   // index of quadrature point

            LineProducingSpecies &lspec = model.lines.lineProducingSpecies[l];

            const double freq_line = lspec.linedata.frequency[k];
            const double invr_mass = lspec.linedata.inverse_mass;
            const double constante = lspec.linedata.A[k] * lspec.quadrature.weights[z] * w_ang;

            add_L_diag (thermodyn, invr_mass, freq_line, constante, rp, f, k, lspec.lambda);

            for (long m = 0; (m < n_off_diag) && (m < n_tot[rp]-1); m++)
            {
                if (n1[rp] >= m+1)   // n1[rp]-m-1 >= 0
                {
                    add_L_lower (thermodyn, invr_mass, freq_line, constante, rp, f, k, m, lspec.lambda);
                }

                if (n1[rp]+m+2+m < n_tot[rp])   // n1[rp]+m+1 < n_tot[rp]-1-m
                {
                    add_L_upper (thermodyn, invr_mass, freq_line, constante, rp, f, k, m, lspec.lambda);
                }
            }
        }
    }
}



///  Store the result of the solver in the model
///    @param[in/out] model : model object under consideration
//////////////////////////////////////////////////////////////

template <typename Real>
inline void Solver<Real> :: store (Model &model) //const
{
    for (Size rp = 0; rp < nraypairs; rp++)
    {
        const double weight_ang = 2.0 * model.geometry.rays.weight(origins[rp], rr);

        const Size i0 = model.radiation.index(origins[rp], 0);

        for (Size f = 0; f < nfreqs; f++)
        {
            model.radiation.J[i0+f] += weight_ang * Su[I(n1[rp], V(rp,f))];
        }

        if (model.parameters.use_scattering())
        {
            for (Size f = 0; f < nfreqs; f++)
            {
                model.radiation.u[RR][i0+f] = Su[I(n1[rp], V(rp,f))];
//                model.radiation.v[RR][i0+f] = Sv[I(n1[rp], V(rp,f))] * reverse[rp];
            }
        }
    }

    update_Lambda (model);
}




///  Gaussian line profile function
///    @param[in] width     : profile width
///    @param[in] freq_diff : frequency difference with line centre
///    @return profile function evaluated with this frequency difference
////////////////////////////////////////////////////////////////////////

template <typename Real>
HOST_DEVICE
inline Real Solver<Real> :: gaussian (const Real width, const Real diff) const
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

template <typename Real>
HOST_DEVICE
inline Real Solver<Real> :: planck (const Real temperature, const Real frequency) const
{
    return TWO_HH_OVER_CC_SQUARED * (frequency*frequency*frequency)
           / expm1 (HH_OVER_KB*frequency/temperature);
}




///  Getter for the emissivity (eta) and the opacity (chi)
///    @param[in]     freq_scaled : frequency (in co-moving frame)
///    @param[in]     p           : in dex of the cell
///    @param[in/out] lnotch      : notch indicating location in the set of lines
///    @param[out]    eta         : emissivity
///    @param[out]    chi         : opacity
/////////////////////////////////////////////////////////////////////////////////

template <typename Real>
HOST_DEVICE
inline void Solver<Real> :: get_eta_and_chi (const Size Dn, const Real frequency, Real &eta, Real &chi)
{
    const Real frequency_scaled = frequency * shifts[Dn];

    /// Initialize
    eta = 0.0E+00;
    chi = 0.0E+00; //1.0E-26;

    /// Set line emissivity and opacity
    for (Size l = 0; l < nlines; l++)
    {
        const Size lnl     = L(nrs[Dn], l);
        const Real diff    = frequency_scaled - (Real) line[l];
        const Real profile = frequency_scaled * gaussian (line_width[lnl], diff);

        eta = my_fma (profile, line_emissivity[lnl], eta);
        chi = my_fma (profile, line_opacity   [lnl], chi);
    }
}




///  Getter for the emissivity (eta) and the opacity (chi)
///    @param[in]     freq_scaled : frequency (in co-moving frame)
///    @param[in]     p           : in dex of the cell
///    @param[in/out] lnotch      : notch indicating location in the set of lines
///    @param[out]    eta         : emissivity
///    @param[out]    chi         : opacity
/////////////////////////////////////////////////////////////////////////////////

template <typename Real>
HOST_DEVICE
inline void Solver<Real> :: get_eta_and_chi (const Size In, const Size Dn, const Real frequency)
{
    const Real frequency_scaled = frequency * shifts[Dn];

    /// Initialize
    eta[In] = 0.0E+00;
    chi[In] = 0.0E+00; //1.0E-26;

    /// Set line emissivity and opacity
    for (Size l = 0; l < nlines; l++)
    {
        const Size lnl     = L(nrs[Dn], l);
        const Real diff    = frequency_scaled - (Real) line[l];
        const Real profile = frequency_scaled * gaussian (line_width[lnl], diff);

        eta[In] = my_fma (profile, line_emissivity[lnl], eta[In]);
        chi[In] = my_fma (profile, line_opacity   [lnl], chi[In]);
    }
}




///  Solver for Feautrier equation along ray pairs using the (ordinary)
///  2nd-order solver, without adaptive optical depth increments
///    @param[in] w : width index
///////////////////////////////////////////////////////////////////////

template <typename Real>
HOST_DEVICE
inline void Solver<Real> :: solve_2nd_order_Feautrier_non_adaptive (const Size w)
{
    // Get indices of the block
    const Size rp = w / nfreqs_red;   // raypair index
    const Size f  = w % nfreqs_red;   // frequency index

    // Get frequency value corresponding to the block
    const Real frequency = frequencies[V(origins[rp], f)];

    // Get indices for first element in the block
    const Size If   = I(first,   w);
    const Size Ifp1 = I(first+1, w);
    const Size Df   = D(rp, first);

    // Get optical properties for first two elements
    get_eta_and_chi (If,   Df,   frequency);
    get_eta_and_chi (Ifp1, Df+1, frequency);

    term1[If  ] = eta[If]   / chi[If];
    term1[Ifp1] = eta[Ifp1] / chi[Ifp1];

    // Get first optical depth increment
    dtau[If] = 0.5 * (chi[If] + chi[Ifp1]) * dZs[Df];

    // Set boundary conditions
    const Real inverse_dtau0 = one / dtau[If];

    C[If] = 2.0 * inverse_dtau0 * inverse_dtau0;

    const Real B0_min_C0 = my_fma (2.0, inverse_dtau0, one);
    const Real B0        = B0_min_C0 + C[If];

    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[Df]);

    Su[If] = term1[If] + 2.0 * I_bdy_0 * inverse_dtau0;

    /// Start elimination step
    Su[If] = Su[If] / B0;

    /// Write economically: F[first] = (B[first] - C[first]) / C[first];
    F[If] = 0.5 * B0_min_C0 * dtau[If] * dtau[If];
    inverse_one_plus_F[If] = one / (one + F[If]);


    /// Set body of Feautrier matrix
    for (Size n = first+1; n < last; n++)
    {
        const Size Inm1  = I(n-1, w);
        const Size In    = I(n,   w);
        const Size Inp1  = I(n+1, w);
        const Size Dn    = D(rp,  n);

        // Get new optical properties
        get_eta_and_chi (Inp1, Dn+1, frequency);

        // Compute term1 at n+1
        term1[Inp1] = eta[Inp1] / chi[Inp1];

        // Compute the optical depth increment
        dtau[In] = 0.5 * (chi[In] + chi[Inp1]) * dZs[Dn];

        const Real dtau_avg = 0.5 * (dtau[Inm1] + dtau[In]);
        inverse_A[In] = dtau_avg * dtau[Inm1];
        inverse_C[In] = dtau_avg * dtau[In  ];

        A[In] = one / inverse_A[In];
        C[In] = one / inverse_C[In];

        /// Use the previously stored value of the source function
        Su[In] = term1[In];


        F[In] = my_fma (A[In]*F[Inm1], inverse_one_plus_F[Inm1], one) * inverse_C[In];
        inverse_one_plus_F[In] = one / (one + F[In]);

        Su[In] = my_fma (A[In], Su[Inm1], Su[In]) * inverse_one_plus_F[In] * inverse_C[In];
    }


    // Get indices for first element in the block
    const Size Il   = I(last,   w);
    const Size Ilm1 = I(last-1, w);
    const Size Dl   = D(rp, last);

    /// Set boundary conditions
    const Real inverse_dtaud = one / dtau[Ilm1];

    A[Il] = 2.0 * inverse_dtaud * inverse_dtaud;

    const Real Bd_min_Ad = my_fma (2.0, inverse_dtaud, one);
    const Real Bd        = Bd_min_Ad + A[Il];

    const Real denominator = one / my_fma (Bd, F[Ilm1], Bd_min_Ad);

    const Real I_bdy_n = planck (T_CMB, frequency*shifts[Dl]);

    Su[Il] = term1[Il] + 2.0 * I_bdy_n * inverse_dtaud;
    Su[Il] = my_fma (A[Il], Su[Ilm1], Su[Il]) * (one + F[Ilm1]) * denominator;


    if (n_off_diag == 0)
    {
        if (n1_min < last)
        {
            /// Write economically: G[last] = (B[last] - A[last]) / A[last];
            G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
            G_over_one_plus_G[Il] = G[Il] / (one + G[Il]);

            for (long n = last-1; n > n1_min; n--) // use long in reverse loops!
            {
                const Size Inp1 = I(n+1, w);
                const Size In   = I(n,   w);

                Su[In] = my_fma(Su[Inp1], inverse_one_plus_F[In], Su[In]);

                                G[In] = my_fma(C[In], G_over_one_plus_G[Inp1], one) * inverse_A[In];
                G_over_one_plus_G[In] = G[In] / (one + G[In]);
            }

            const Size In1   = I(n1_min,   w);
            const Size In1p1 = I(n1_min+1, w);

                Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
            L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);

//            printf("- Solver: first %ld, last %ld, n_tot %ld, r %ld\n", first, last, n_tot[rp], rr);
//            printf("L(n=%ld, f=%ld) = %le\n", n1_min, f, L_diag[In1]);
        }
        else
        {
            const Size In1   = I(n1_min,   w);
            const Size In1m1 = I(n1_min-1, w);

            L_diag[In1] = (one + F[In1m1]) / (Bd_min_Ad + Bd*F[In1m1]);
//            printf("- Solver: first %ld, last %ld, n_tot %ld, r %ld\n", first, last, n_tot[rp], rr);
//            printf("L(n=%ld, f=%ld) = %le\n", n1_min, f, L_diag[In1]);
        }
    }
    else
    {
        /// Write economically: G[last] = (B[last] - A[last]) / A[last];
                         G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
        inverse_one_plus_G[Il] = one / (one + G[Il]);
         G_over_one_plus_G[Il] = G[Il] * inverse_one_plus_G[Il];

        L_diag[Il] = (one + F[Ilm1]) / (Bd_min_Ad + Bd*F[Ilm1]);

        for (long n = last-1; n > first; n--) // use long in reverse loops!
        {
            const Size Inp1 = I(n+1, w);
            const Size In   = I(n,   w);

            Su[In] = my_fma(Su[Inp1], inverse_one_plus_F[In], Su[In]);

                             G[In] = my_fma(C[In], G_over_one_plus_G[Inp1], one) * inverse_A[In];
            inverse_one_plus_G[In] = one / (one + G[In]);
             G_over_one_plus_G[In] = G[In] * inverse_one_plus_G[In];

            L_diag[In] = inverse_C[In] / (F[In] + G_over_one_plus_G[Inp1]);
//            printf("L(n=%ld, f=%ld) = %le\n", n, f, L_diag[In]);
        }

            Su[If] = my_fma(Su[Ifp1], inverse_one_plus_F[If], Su[If]);
        L_diag[If] = (one + G[Ifp1]) / (B0_min_C0 + B0*G[Ifp1]);

//        printf("first G[Ifp1] = %le\n", G[Ifp1]);
//        printf("L(n=%ld, f=%ld) = %le\n", first, f, L_diag[If]);


        for (long n = last-1; n >= first; n--) // use long in reverse loops!
        {
            const Size Inp1 = I(n+1, w);
            const Size In   = I(n,   w);

            L_upper[M(0,In)] = L_diag[Inp1] * inverse_one_plus_F[In  ];
            L_lower[M(0,In)] = L_diag[In  ] * inverse_one_plus_G[Inp1];

//            printf("L_u(0, %ld) = %le\n", In, L_upper[M(0,In)]);
//            printf("L_l(0, %ld) = %le\n", In, L_lower[M(0,In)]);
        }

        for (Size m = 1; (m < n_off_diag) && (m < n_tot[rp]-1); m++)
        {
            for (long n = last-1-m; n >= first; n--) // use long in reverse loops!
            {
                const Size Inp1   = I(n+1,   w);
                const Size Inpmp1 = I(n+m+1, w);
                const Size In     = I(n,     w);

                L_upper[M(m,In)] = L_upper[M(m-1,Inp1)] * inverse_one_plus_F[In    ];
                L_lower[M(m,In)] = L_lower[M(m-1,In  )] * inverse_one_plus_G[Inpmp1];

//                printf("L_u(%ld, %ld)[%ld] = %le\n", m, In, M(m,In), L_upper[M(m,In)]);
//                printf("L_l(%ld, %ld)[%ld] = %le\n", m, In, M(m,In), L_lower[M(m,In)]);
            }
        }

//        for (long n = last-1; n >= first; n--) // use long in reverse loops!
//        {
//            const Size Inp1 = I(n+1, w);
//            const Size In   = I(n,   w);
//
//            printf("check L_u(0, %ld)[%ld] = %le\n", In, M(0,In), L_upper[M(0,In)]);
//            printf("check L_l(0, %ld)[%ld] = %le\n", In, M(0,In), L_lower[M(0,In)]);
//        }
    }

}




///  Solver for Feautrier equation along ray pairs using the (ordinary)
///  2nd-order solver, with adaptive optical depth increments
///    @param[in] w : width index
///////////////////////////////////////////////////////////////////////
///   NOT PROPERLY IMPLEMENTED YET!!!
///////////////////////////////////////////////////////////////////////

template <typename Real>
HOST_DEVICE
inline void Solver<Real> :: solve_2nd_order_Feautrier_adaptive (const Size w)
{
    // Get indices of the block
    const Size rp = w / nfreqs_red;   // raypair index
    const Size f  = w % nfreqs_red;   // frequency index

    // Get frequency value corresponding to the block
    const Real frequency = frequencies[V(origins[rp], f)];

    // Get indices for first element in the block
    const Size If   = I(first,   w);
    const Size Ifp1 = I(first+1, w);
    const Size Df   = D(rp, first);


    // Get optical properties for first element
    Real eta_n, chi_n;
    get_eta_and_chi (Df,   frequency, eta_n, chi_n);
    Real tm1_n = eta_n / chi_n;

    // Get optical properties for second element
    Real eta_1, chi_1;
    get_eta_and_chi (Df+1, frequency, eta_1, chi_1);
    Real tm1_1 = eta_1 / chi_1;

    // Get first optical depth increment
    dtau[If] = 0.5 * (chi_n + chi_1) * dZs[Df];

    // Set boundary conditions
    const Real inverse_dtau0 = one / dtau[If];

    C[If] = 2.0 * inverse_dtau0 * inverse_dtau0;

    const Real B0_min_C0 = my_fma (2.0, inverse_dtau0, one);
    const Real B0        = B0_min_C0 + C[If];

    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[Df]);

    Su[If] = tm1_n + 2.0 * I_bdy_0 * inverse_dtau0;

    /// Start elimination step
    Su[If] = Su[If] / B0;

    /// Write economically: F[first] = (B[first] - C[first]) / C[first];
    F[If] = 0.5 * B0_min_C0 * dtau[If] * dtau[If];
    inverse_one_plus_F[If] = one / (one + F[If]);


    Size Inm1  = If;
    Size In    = Ifp1;
    Size Inp1  = Ifp1 + width;
    Size Dn    = Df+1;

    /// Set body of Feautrier matrix
    for (Size n = first+1; n < last; n++)
    {
        // Get new optical properties
        eta_n = eta_1;
        chi_n = chi_1;
        get_eta_and_chi (Dn+1, frequency, eta_1, chi_1);

        // Compute term1 at n+1
        tm1_n = tm1_1;
        tm1_1 = eta_1 / chi_1;

        // Compute the optical depth increment
        dtau[In] = 0.5 * (chi_n + chi_1) * dZs[Dn];


        const Real dtau_avg = 0.5 * (dtau[Inm1] + dtau[In]);
        inverse_A[In] = dtau_avg * dtau[Inm1];
        inverse_C[In] = dtau_avg * dtau[In  ];

        A[In] = one / inverse_A[In];
        C[In] = one / inverse_C[In];

        /// Use the previously stored value of the source function
        Su[In] = tm1_n;


        F[In] = my_fma (A[In]*F[Inm1], inverse_one_plus_F[Inm1], one) * inverse_C[In];
        inverse_one_plus_F[In] = one / (one + F[In]);

        Su[In] = my_fma (A[In], Su[Inm1], Su[In]) * inverse_one_plus_F[In] * inverse_C[In];

        Inm1  = In;
        In    = Inp1;
        Inp1 += width;
        Dn++;
    }


    // Get indices for first element in the block
    const Size Il   = I(last,   w);
    const Size Ilm1 = I(last-1, w);
    const Size Dl   = D(rp, last);

    /// Set boundary conditions
    const Real inverse_dtaud = one / dtau[Ilm1];

    A[Il] = 2.0 * inverse_dtaud * inverse_dtaud;

    const Real Bd_min_Ad = my_fma (2.0, inverse_dtaud, one);
    const Real Bd        = Bd_min_Ad + A[Il];

    const Real denominator = one / my_fma (Bd, F[Ilm1], Bd_min_Ad);

    const Real I_bdy_n = planck (T_CMB, frequency*shifts[Dl]);

    Su[Il] = tm1_1 + 2.0 * I_bdy_n * inverse_dtaud;
    Su[Il] = my_fma (A[Il], Su[Ilm1], Su[Il]) * (one + F[Ilm1]) * denominator;


    if (n_off_diag == 0)
    {
        if (n1_min < last)
        {
            /// Write economically: G[last] = (B[last] - A[last]) / A[last];
                            G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
            G_over_one_plus_G[Il] = G[Il] / (one + G[Il]);

            for (Size n = last-1; n > n1_min; n--) // use long in reverse loops!
            {
                const Size Inp1 = I(n+1, w);
                const Size In   = I(n,   w);

                Su[In] = my_fma(Su[Inp1], inverse_one_plus_F[In], Su[In]);

                                G[In] = my_fma(C[In], G_over_one_plus_G[Inp1], one) * inverse_A[In];
                G_over_one_plus_G[In] = G[In] / (one + G[In]);
            }

            const Size In1   = I(n1_min,   w);
            const Size In1p1 = I(n1_min+1, w);

                Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
            L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);
        }
        else
        {
            const Size In1   = I(n1_min,   w);
            const Size In1m1 = I(n1_min-1, w);

            L_diag[In1] = (one + F[In1m1]) / (Bd_min_Ad + Bd*F[In1m1]);
        }
    }
    else
    {
        /// Write economically: G[last] = (B[last] - A[last]) / A[last];
                         G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
        inverse_one_plus_G[Il] = one / (one + G[Il]);
         G_over_one_plus_G[Il] = G[Il] * inverse_one_plus_G[Il];

        L_diag[Il] = (one + F[Ilm1]) / (Bd_min_Ad + Bd*F[Ilm1]);

        for (long n = last-1; n > first; n--) // use long in reverse loops!
        {
            const Size Inp1 = I(n+1, w);
            const Size In   = I(n,   w);

            Su[In] = my_fma(Su[Inp1], inverse_one_plus_F[In], Su[In]);

                             G[In] = my_fma(C[In], G_over_one_plus_G[Inp1], one) * inverse_A[In];
            inverse_one_plus_G[In] = one / (one + G[In]);
             G_over_one_plus_G[In] = G[In] * inverse_one_plus_G[In];

            L_diag[In] = inverse_C[In] / (F[In] + G_over_one_plus_G[Inp1]);
        }

            Su[If] = my_fma(Su[Ifp1], inverse_one_plus_F[If], Su[If]);
        L_diag[If] = (one + G[Ifp1]) / (B0_min_C0 + B0*G[Ifp1]);


        for (long n = last-1; n >= first; n--) // use long in reverse loops!
        {
            const Size Inp1 = I(n+1, w);
            const Size In   = I(n,   w);

//            cout << endl;
//            cout << "first   = " << first   << endl;
//            cout << "n       = " << n       << endl;
//
//            cout << "w       = " << w       << endl;
//            cout << "In      = " << In      << endl;
//            cout << "Inp1    = " << Inp1    << endl;
//            cout << "M(0,In) = " << M(0,In) << endl;
//            cout << "iopF    = " << inverse_one_plus_F[In]   << endl;
//            cout << "iopG    = " << inverse_one_plus_G[Inp1] << endl;

            L_upper[M(0,In)] = L_diag[Inp1] * inverse_one_plus_F[In  ];
            L_lower[M(0,In)] = L_diag[In  ] * inverse_one_plus_G[Inp1];
        }

        for (Size m = 1; (m < n_off_diag) && (m < n_tot[rp]-1); m++)
        {
            for (long n = last-1-m; n >= first; n--) // use long in reverse loops!
            {
                const Size Inp1   = I(n+1,   w);
                const Size Inpmp1 = I(n+m+1, w);
                const Size In     = I(n,     w);

                L_upper[M(m,In)] = L_upper[M(m-1,Inp1)] * inverse_one_plus_F[In    ];
                L_lower[M(m,In)] = L_lower[M(m-1,In  )] * inverse_one_plus_G[Inpmp1];
            }
        }
    }

}




///  Solver for Feautrier equation along ray pairs using the (Hermitian)
///  4th-order solver, without adaptive optical depth increments
///    @param[in] w : width index
////////////////////////////////////////////////////////////////////////

template <typename Real>
HOST_DEVICE
inline void Solver<Real> :: solve_4th_order_Feautrier_non_adaptive (const Size w)
{
    // Get indices of the block
    const Size rp = w / nfreqs_red;   // raypair index
    const Size f  = w % nfreqs_red;   // frequency index

    // Get frequency value corresponding to the block
    const Real frequency = frequencies[V(origins[rp], f)];

    // Get indices for first element in the block
    const Size If   = I(first,   w);
    const Size Ifp1 = I(first+1, w);
    const Size Df   = D(rp, first);

    // Get optical properties for first element
    Real eta_n, chi_n;
    get_eta_and_chi (Df,   frequency, eta_n, chi_n);
    Real tm1_n = eta_n / chi_n;

    // Get optical properties for second element
    Real eta_1, chi_1;
    get_eta_and_chi (Df+1, frequency, eta_1, chi_1);
    Real tm1_1 = eta_1 / chi_1;

    // Get optical properties for second element
    dtau[If] = 0.5 * (chi_n + chi_1) * dZs[Df];

    // Set boundary conditions
    const Real inverse_dtau0 = one / dtau[If];

    C[If] = 2.0 * inverse_dtau0 * inverse_dtau0 - ONE_THIRD;

    const Real B0_min_C0 = my_fma (2.0, inverse_dtau0, one);
    const Real B0        = B0_min_C0 + C[If];

    const Real I_bdy_0 = planck (T_CMB, frequency*shifts[Df]);

    Su[If] = 2.0 * I_bdy_0 * inverse_dtau0;
    Su[If] = my_fma (ONE_THIRD,  tm1_1, Su[If]);
    Su[If] = my_fma (TWO_THIRDS, tm1_n, Su[If]);

    /// Start elimination step
    Su[If] = Su[If] / B0;

    /// Write economically: F[first] = (B[first] - C[first]) / C[first];
    F[If] = 0.5 * B0_min_C0 * dtau[If] * dtau[If];
    inverse_one_plus_F[If] = one / (one + F[If]);


    Size Inm1  = If;
    Size In    = Ifp1;
    Size Inp1  = Ifp1 + width;
    Size Dn    = Df+1;
    Size index = first+1;

    /// Set body of Feautrier matrix
    for (Size n = first+1; n < last; n++)
    {
        // Get new optical properties
        eta_n = eta_1;
        chi_n = chi_1;
        get_eta_and_chi (Dn+1, frequency, eta_1, chi_1);

        /// Store the previous value of the source function
        Su[In] = tm1_n;

        // Compute term1 at n+1
        tm1_n = tm1_1;
        tm1_1 = eta_1 / chi_1;

        // Compute the optical depth increment
        dtau[In] = 0.5 * (chi_n + chi_1) * dZs[Dn];


        const Real dtau_tot = dtau[Inm1] + dtau[In];
        A[In] = one / (dtau_tot * dtau[Inm1]);
        C[In] = one / (dtau_tot * dtau[In  ]);

        a[In] = ONE_SIXTH * my_fma (-A[In], dtau[In  ] * dtau[In  ], one);
        c[In] = ONE_SIXTH * my_fma (-C[In], dtau[Inm1] * dtau[Inm1], one);

        A[In] = my_fma (2.0, A[In], -a[In]);
        C[In] = my_fma (2.0, C[In], -c[In]);

        inverse_A[In] = one / A[In];
        inverse_C[In] = one / C[In];

        /// Use the previously stored value of the source function
        Su[In] = Su[In] * a[In];
        Su[In] = my_fma (one - a[In] - c[In], tm1_n, Su[In]);
        Su[In] = my_fma (              c[In], tm1_1, Su[In]);


        F[In] = my_fma (A[In]*F[Inm1], inverse_one_plus_F[Inm1], one) * inverse_C[In];
        inverse_one_plus_F[In] = one / (one + F[In]);

        Su[In] = my_fma (A[In], Su[Inm1], Su[In]) * inverse_one_plus_F[In] * inverse_C[In];

        Inm1  = In;
        In    = Inp1;
        Inp1 += width;
        index++;
        Dn++;
    }


    // Get indices for first element in the block
    const Size Il   = I(index,   w);
    const Size Ilm1 = I(index-1, w);
    const Size Dl   = D(rp, last);

    /// Set boundary conditions
    const Real inverse_dtaud = one / dtau[Ilm1];

    A[Il] = 2.0 * inverse_dtaud * inverse_dtaud - ONE_THIRD;

    const Real Bd_min_Ad = my_fma (2.0, inverse_dtaud, one);
    const Real Bd        = Bd_min_Ad + A[Il];

    const Real denominator = one / my_fma (Bd, F[Ilm1], Bd_min_Ad);

    const Real I_bdy_n = planck (T_CMB, frequency*shifts[Dl]);

    Su[Il] = 2.0 * I_bdy_n * inverse_dtaud;
    Su[Il] = my_fma (ONE_THIRD,  tm1_n, Su[Il]);
    Su[Il] = my_fma (TWO_THIRDS, tm1_1, Su[Il]);
    Su[Il] = my_fma (A[Il], Su[Ilm1], Su[Il]) * (one + F[Ilm1]) * denominator;


    if (n1_min < index)
    {
        /// Write economically: G[last] = (B[last] - A[last]) / A[last];
        G[Il] = 0.5 * Bd_min_Ad * dtau[Ilm1] * dtau[Ilm1];
        G_over_one_plus_G[Il] = G[Il] / (one + G[Il]);

        for (Size n = index-1; n > n1_min; n--)
        {
            const Size Inp1 = I(n+1, w);
            const Size In   = I(n,   w);

            Su[In] = my_fma (Su[Inp1], inverse_one_plus_F[In  ], Su[In]);
             G[In] = my_fma ( C[In  ],  G_over_one_plus_G[Inp1], one   ) * inverse_A[In];

            G_over_one_plus_G[In] = G[In] / (one + G[In]);
        }

        const Size In1   = I(n1_min,   w);
        const Size In1p1 = I(n1_min+1, w);

            Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
        L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);
    }
    else
    {
        const Size In1   = I(n1_min,   w);
        const Size In1m1 = I(n1_min-1, w);

        L_diag[In1] = (one + F[In1m1]) / (Bd_min_Ad + Bd*F[In1m1]);
    }

}
