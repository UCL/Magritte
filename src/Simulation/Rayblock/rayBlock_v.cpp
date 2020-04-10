#include "rayBlock_v.hpp"
#include "Simulation/simulation.hpp"


///  Constructor for RayBlock_v
///    @param[in] ncells    : number of cells
///    @param[in] nfreqs    : number of frequency bins
///    @param[in] nlines    : number of lines
///    @param[in] nraypairs : number of ray pairs
///    @param[in] depth     : number of points along the ray pairs
//////////////////////////////////////////////////////////////////

RayBlock_v :: RayBlock_v (
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

    n1 = (Size*) malloc (nraypairs_size);
    n2 = (Size*) malloc (nraypairs_size);

    origins = (Size*) malloc (nraypairs_size);
    reverse = (Real*) malloc (nraypairs_real);

    nrs    = (Size*) malloc (depth_max*nraypairs_max*sizeof(Size));
    shifts = (Real*) malloc (depth_max*nraypairs_max*sizeof(Real));
    dZs    = (Real*) malloc (depth_max*nraypairs_max*sizeof(Real));

    line            = (double*) malloc (       nlines*sizeof(double));
    line_emissivity = (double*) malloc (ncells*nlines*sizeof(double));
    line_opacity    = (double*) malloc (ncells*nlines*sizeof(double));
    line_width      = (double*) malloc (ncells*nlines*sizeof(double));

    frequencies     = (double*) malloc (ncells*nfreqs*sizeof(double));

    const size_t area_real = depth_max*width_max*sizeof(Real);
    const size_t area_size = depth_max*width_max*sizeof(Size);

    term1              =  (Real*) malloc (area_real);
    term2              =  (Real*) malloc (area_real);

    eta                =  (Real*) malloc (area_real);
    chi                =  (Real*) malloc (area_real);

    A                  =  (Real*) malloc (area_real);
    a                  =  (Real*) malloc (area_real);
    C                  =  (Real*) malloc (area_real);
    c                  =  (Real*) malloc (area_real);
    F                  =  (Real*) malloc (area_real);
    G                  =  (Real*) malloc (area_real);

    inverse_A          =  (Real*) malloc (area_real);
    inverse_C          =  (Real*) malloc (area_real);
    inverse_one_plus_F =  (Real*) malloc (area_real);
    inverse_one_plus_G =  (Real*) malloc (area_real);
     G_over_one_plus_G =  (Real*) malloc (area_real);

    Su                 =  (Real*) malloc (area_real);
    Sv                 =  (Real*) malloc (area_real);
    dtau               =  (Real*) malloc (area_real);

    L_diag             =  (Real*) malloc (area_real);
}




///  Destructor for gpuRayPair
//////////////////////////////

RayBlock_v :: ~RayBlock_v ()
{
    free (n1);
    free (n2);

    free (origins);
    free (reverse);

    free (nrs);
    free (shifts);
    free (dZs);

    free (line);
    free (line_emissivity);
    free (line_opacity);
    free (line_width);

    free (frequencies);

    free (term1);
    free (term2);

    free (eta);
    free (chi);

    free (A);
    free (a);
    free (C);
    free (c);
    free (F);
    free (G);

    free (inverse_A);
    free (inverse_C);
    free (inverse_one_plus_F);
    free (inverse_one_plus_G);
    free ( G_over_one_plus_G);

    free (Su);
    free (Sv);
    free (dtau);

    free (L_diag);
}




///  Copies the model data into the gpuRayPair data structure
///    @param[in] model : model from which to copy
/////////////////////////////////////////////////////////////

void RayBlock_v :: copy_model_data (const Model &model)
{
    memcpy (line,
            model.lines.line.data(),
            model.lines.line.size()*sizeof(double));
    memcpy (line_emissivity,
            model.lines.emissivity.data(),
            model.lines.emissivity.size()*sizeof(double));
    memcpy (line_opacity,
            model.lines.opacity.data(),
            model.lines.opacity.size()*sizeof(double));


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


void RayBlock_v :: setup (
        const Model           &model,
        const Size             R,
        const Size             r,
        const ProtoRayBlock_v &prb   )
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


inline Real my_fma (const Real a, const Real b, const Real c)
{
    return a * b + c;
}


void feautrierKernel (RayBlock_v &rayblock)
{
    for (Size w = 0; w < rayblock.width; w++)
    {
        rayblock.solve_Feautrier (w);
    }
}


void RayBlock_v :: solve ()
{
    feautrierKernel (*this);
}




void RayBlock_v :: store (Model &model) const
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

void RayBlock_v :: get_eta_and_chi (const Size Dn, const Real frequency, Real &eta, Real &chi)
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




///  Solver for the Feautrier equation along the ray pair
///    @param[in] w : width index
/////////////////////////////////////////////////////////

void RayBlock_v :: solve_Feautrier (const Size w)
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

        /// Get the maximum chi value
        Real chi_max;
        if (chi_n < chi_1) {chi_max = chi_1;}
        else               {chi_max = chi_n;}

        /// Get the number of interpolations
        Size n_interpl = chi_max * dZs[Dn] * inverse_dtau_max + 1;

        /// Limit the number of interpolations
        if (n_interpl > 10) {n_interpl = 10;}

        /// Invert the number of interpolations
        const Real inverse_n = 1.0 / n_interpl;

        /// Get the index of the result
        if (n1[rp] == n) {n1_local = index;}

        /// Prepare the interpolation variables
        const Real ldZs = dZs[Dn]         * inverse_n;
        const Real leta = (eta_1 - eta_n) * inverse_n;
        const Real lchi = (chi_1 - chi_n) * inverse_n;

        printf("n_interpl = %ld\n", n_interpl);

        for (Size i = 1; i <= n_interpl; i++)
        {
            /// Linearly interpolate the emissivity and the opacity
            eta_1 = my_fma (i, leta, eta_n);
            chi_1 = my_fma (i, lchi, chi_n);

            /// Store the previous value of the source function
            Su[In] = tm1_n;

            /// Compute term1 at n+1
            tm1_n = tm1_1;
            tm1_1 = eta_1 / chi_1;

            /// Compute the optical depth increment
            dtau[In] = 0.5 * (chi_n + chi_1) * ldZs;

            const Real dtau_tot = dtau[Inm1] + dtau[In];

            const Real AA = 1.0 / (dtau_tot * dtau[Inm1]);
            const Real CC = 1.0 / (dtau_tot * dtau[In  ]);

            a[In] = ONE_SIXTH * my_fma (-AA, dtau[In  ] * dtau[In  ], 1.0);
            c[In] = ONE_SIXTH * my_fma (-CC, dtau[Inm1] * dtau[Inm1], 1.0);

            A[In] = my_fma (2.0, AA, -a[In]);
            C[In] = my_fma (2.0, CC, -c[In]);

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
        }

        Dn++;
    }


    /// Get the index of the result
    if (n1[rp] == last) {n1_local = index;}


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
         G[In] = my_fma ( C[In  ],  G_over_one_plus_G[Inp1],    1.0) * inverse_A[In];

        G_over_one_plus_G[In] = G[In] / (1.0 + G[In]);
    }


    const Size In1   = I(n1_min,   w);
    const Size In1p1 = I(n1_min+1, w);

    Su[In1] = my_fma (Su[In1p1], inverse_one_plus_F[In1], Su[In1]);
    L_diag[In1] = inverse_C[In1] / (F[In1] + G_over_one_plus_G[In1p1]);


    // TODO: Avoid this remapping.

    /// Map the result to the expected place
    Su[I(n1[rp],w)] = Su[I(n1_local,w)];

}
