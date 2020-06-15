#include "simd_solver.hpp"
#include "Simulation/simulation.hpp"


///  Constructor for cpuSolver
///    @param[in] ncells    : number of cells
///    @param[in] nfreqs    : number of frequency bins
///    @param[in] nlines    : number of lines
///    @param[in] nraypairs : number of ray pairs
///    @param[in] depth     : number of points along the ray pairs
//////////////////////////////////////////////////////////////////

simdSolver :: simdSolver (
    const Size ncells,
    const Size nfreqs,
    const Size nlines,
    const Size nraypairs,
    const Size depth,
    const Size n_off_diag )
    : Solver (ncells, nfreqs, reduced(nfreqs), nlines, nraypairs, depth, n_off_diag)
{
    cout << endl << "############################" << endl << endl;
    cout << "nsimd = " << n_simd_lanes << endl;
    cout << endl << "############################" << endl << endl;


    n1 = new size_t[nraypairs_max];
    n2 = new size_t[nraypairs_max];

    n_tot = new size_t[nraypairs_max];

    origins = new size_t[nraypairs_max];
    reverse = new double[nraypairs_max];

    nrs    = new size_t[depth_max*nraypairs_max];
    shifts = new double[depth_max*nraypairs_max];
    dZs    = new double[depth_max*nraypairs_max];

    line            = new double[       nlines];
    line_emissivity = new double[ncells*nlines];
    line_opacity    = new double[ncells*nlines];
    line_width      = new double[ncells*nlines];

    frequencies = new vReal[ncells*nfreqs_red];

    const Size area = 10*depth_max*width_max;

    term1              = new vReal[area];
    term2              = new vReal[area];

    eta                = new vReal[area];
    chi                = new vReal[area];

    A                  = new vReal[area];
    a                  = new vReal[area];
    C                  = new vReal[area];
    c                  = new vReal[area];
    F                  = new vReal[area];
    G                  = new vReal[area];

    inverse_A          = new vReal[area];
    inverse_C          = new vReal[area];
    inverse_one_plus_F = new vReal[area];
    inverse_one_plus_G = new vReal[area];
     G_over_one_plus_G = new vReal[area];

    Su                 = new vReal[area];
    Sv                 = new vReal[area];
    dtau               = new vReal[area];

    L_diag             = new vReal[area];

    if (n_off_diag > 0)
    {
        L_upper = new vReal[n_off_diag*area];
        L_lower = new vReal[n_off_diag*area];
    }
}


///  Destructor for cpuSolver
/////////////////////////////

simdSolver :: ~simdSolver ()
{
    delete[] n1;
    delete[] n2;

    delete[] n_tot;

    delete[] origins;
    delete[] reverse;

    delete[] nrs;
    delete[] shifts;
    delete[] dZs;

    delete[] line;
    delete[] line_emissivity;
    delete[] line_opacity;
    delete[] line_width;

    delete[] frequencies;

    delete[] term1;
    delete[] term2;

    delete[] eta;
    delete[] chi;

    delete[] A;
    delete[] a;
    delete[] C;
    delete[] c;
    delete[] F;
    delete[] G;

    delete[] inverse_A;
    delete[] inverse_C;
    delete[] inverse_one_plus_F;
    delete[] inverse_one_plus_G;
    delete[]  G_over_one_plus_G;

    delete[] Su;
    delete[] Sv;
    delete[] dtau;

    delete[] L_diag;
}




///  Copies the model data into the gpuRayPair data structure
///    @param[in] model : model from which to copy
/////////////////////////////////////////////////////////////

void simdSolver :: copy_model_data (const Model &model)
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
            const Size           f = model.lines.line_index[l];
            const Size       lspec = model.radiation.frequencies.corresponding_l_for_spec[f];
            const double invr_mass = model.lines.lineProducingSpecies[lspec].linedata.inverse_mass;

            line_width[L(p,l)] = model.thermodynamics.profile_width (invr_mass, p, model.lines.line[l]);
        }

        for (Size f = 0; f < nfreqs_red; f++)
        {
            for (size_t lane = 0; lane < n_simd_lanes; lane++)
            {
                const size_t ind = f*n_simd_lanes + lane;

                frequencies[V(p,f)].putlane(model.radiation.frequencies.nu[p][ind], lane);
            }
        }
    }

}




void simdSolver :: solve (
    const ProtoBlock &prb,
    const Size        R,
    const Size        r,
          Model      &model )
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




///  Store the result of the solver in the model
///    @param[in/out] model : model object under consideration
//////////////////////////////////////////////////////////////

inline void simdSolver :: store (Model &model) const
{
    for (Size rp = 0; rp < nraypairs; rp++)
    {
        const double weight_ang = 2.0 * model.geometry.rays.weight(origins[rp], rr);

        const Size i0 = model.radiation.index(origins[rp], 0);
        const Size j0 = I(n1[rp], V(rp, 0));

        for (Size f = 0; f < nfreqs_red; f++)
        {
            for (size_t lane = 0; lane < n_simd_lanes; lane++)
            {
                const size_t ind = i0 + f*n_simd_lanes + lane;

                model.radiation.J[ind] += weight_ang * Su[j0+f].getlane(lane);
            }
        }

        if (model.parameters.use_scattering())
        {
            for (Size f = 0; f < nfreqs_red; f++)
            {
                for (size_t lane = 0; lane < n_simd_lanes; lane++)
                {
                    const size_t ind = i0 + f*n_simd_lanes + lane;

                    model.radiation.u[RR][ind] = Su[j0+f].getlane(lane);
//                model.radiation.v[RR][i0+f] = Sv[j0+f] * reverse[rp];
                }
            }
        }
    }
}
