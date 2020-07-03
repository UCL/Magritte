#include "cpu_solver_Eigen.hpp"
#include "Simulation/simulation.hpp"


///  Constructor for cpuSolverEigen
///    @param[in] ncells    : number of cells
///    @param[in] nfreqs    : number of frequency bins
///    @param[in] nlines    : number of lines
///    @param[in] nboundary : number of boundary cells
///    @param[in] nraypairs : number of ray pairs
///    @param[in] depth     : number of points along the ray pairs
//////////////////////////////////////////////////////////////////

cpuSolverEigen :: cpuSolverEigen (
    const Size ncells,
    const Size nfreqs,
    const Size nlines,
    const Size nboundary,
    const Size nraypairs,
    const Size depth,
    const Size n_off_diag        )
    : Solver (ncells, nfreqs, 1, nlines, nboundary, nraypairs, depth, n_off_diag)
{
    n1 = new size_t[nraypairs_max];
    n2 = new size_t[nraypairs_max];

    n_tot = new size_t[nraypairs_max];

    bdy_0 = new size_t[nraypairs_max];
    bdy_n = new size_t[nraypairs_max];

    origins = new size_t[nraypairs_max];
    reverse = new double[nraypairs_max];

    nrs    = new size_t[depth_max*nraypairs_max];
    shifts = new double[depth_max*nraypairs_max];
    dZs    = new double[depth_max*nraypairs_max];

    line            = new double[       nlines];
    line_emissivity = new double[ncells*nlines];
    line_opacity    = new double[ncells*nlines];
    line_width      = new double[ncells*nlines];

    frequencies = new VectorXd[ncells*nfreqs_red];

    for (size_t p = 0; p < ncells*nfreqs_red; p++)
    {
        frequencies[p].resize (nfreqs);
    }

    boundary_condition   = new BoundaryCondition[nboundary];
    boundary_temperature = new double           [nboundary];

    term1              = new VectorXd[area];
    term2              = new VectorXd[area];

    eta                = new VectorXd[area];
    chi                = new VectorXd[area];

    A                  = new VectorXd[area];
    a                  = new VectorXd[area];
    C                  = new VectorXd[area];
    c                  = new VectorXd[area];
    F                  = new VectorXd[area];
    G                  = new VectorXd[area];

    inverse_A          = new VectorXd[area];
    inverse_C          = new VectorXd[area];
    inverse_one_plus_F = new VectorXd[area];
    inverse_one_plus_G = new VectorXd[area];
     G_over_one_plus_G = new VectorXd[area];

    Su                 = new VectorXd[area];
    Sv                 = new VectorXd[area];
    dtau               = new VectorXd[area];

    L_diag             = new VectorXd[area];

    if (n_off_diag > 0)
    {
        L_upper = new VectorXd[n_off_diag*area];
        L_lower = new VectorXd[n_off_diag*area];
    }

    for (size_t p = 0; p < area; p++)
    {
        term1              [p].resize (nfreqs);
        term2              [p].resize (nfreqs);

        eta                [p].resize (nfreqs);
        chi                [p].resize (nfreqs);

        A                  [p].resize (nfreqs);
        a                  [p].resize (nfreqs);
        C                  [p].resize (nfreqs);
        c                  [p].resize (nfreqs);
        F                  [p].resize (nfreqs);
        G                  [p].resize (nfreqs);

        inverse_A          [p].resize (nfreqs);
        inverse_C          [p].resize (nfreqs);
        inverse_one_plus_F [p].resize (nfreqs);
        inverse_one_plus_G [p].resize (nfreqs);
         G_over_one_plus_G [p].resize (nfreqs);

        Su                 [p].resize (nfreqs);
        Sv                 [p].resize (nfreqs);
        dtau               [p].resize (nfreqs);

        L_diag             [p].resize (nfreqs);
    }

    if (n_off_diag > 0)
    {
        for (size_t p = 0; p < n_off_diag*area; p++)
        {
            L_upper[p].resize (nfreqs);
            L_lower[p].resize (nfreqs);
        }
    }
}


///  Destructor for cpuSolverEigen
//////////////////////////////////

cpuSolverEigen :: ~cpuSolverEigen ()
{
    delete[] n1;
    delete[] n2;

    delete[] n_tot;

    delete[] bdy_0;
    delete[] bdy_n;

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

    delete[] boundary_condition;
    delete[] boundary_temperature;

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

    if (n_off_diag > 0)
    {
        delete[] L_upper;
        delete[] L_lower;
    }
}




///  Copies the model data into the gpuRayPair data structure
///    @param[in] model : model from which to copy
/////////////////////////////////////////////////////////////

void cpuSolverEigen :: copy_model_data (const Model &model)
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
    memcpy (boundary_condition,
            model.geometry.boundary.boundary_condition.data(),
            model.geometry.boundary.boundary_condition.size()*sizeof(BoundaryCondition));
    memcpy (boundary_temperature,
            model.geometry.boundary.boundary_temperature.data(),
            model.geometry.boundary.boundary_temperature.size()*sizeof(double));


    for (Size p = 0; p < ncells; p++)
    {
        for (Size l = 0; l < nlines; l++)
        {
            const Size           f = model.lines.line_index[l];
            const Size       lspec = model.radiation.frequencies.corresponding_l_for_spec[f];
            const double invr_mass = model.lines.lineProducingSpecies[lspec].linedata.inverse_mass;

            line_width[L(p,l)] = model.thermodynamics.profile_width (invr_mass, p, model.lines.line[l]);
        }

        for (Size f = 0; f < nfreqs; f++)
        {
            frequencies[p][f] = model.radiation.frequencies.nu[p][f];
        }
    }

}




void cpuSolverEigen :: solve (
    const ProtoBlock &prb,
    const Size        R,
    const Size        r,
          Model      &model  )
{
    // Start timer
    timer.start();

    // Setup the block with the protoblock data
    setup (model, R, r, prb);

    // Execute the Feautrier kernel on the CPU
    for (Size w = 0; w < width; w++)
    {
        solve_Feautrier (w);
//        check_L         (w);
    }

    // Store the result back in the model
    store (model);

    // Stop timer
    timer.stop();
}




///  Store the result of the solver in the model
///    @param[in/out] model : model object under consideration
//////////////////////////////////////////////////////////////

inline void cpuSolverEigen :: store (Model &model) const
{
    for (Size rp = 0; rp < nraypairs; rp++)
    {
        const double weight_ang = 2.0 * model.geometry.rays.weight(origins[rp], rr);

        const Size i0 = model.radiation.index(origins[rp], 0);
        const Size j0 = I(n1[rp], V(rp, 0));

        for (Size f = 0; f < nfreqs; f++)
        {
            model.radiation.J[i0+f] += weight_ang * Su[j0][f];
        }

        if (model.parameters.use_scattering())
        {
            for (Size f = 0; f < nfreqs; f++)
            {
                model.radiation.u[RR][i0+f] = Su[j0][f];
//                model.radiation.v[RR][i0+f] = Sv[j0+f] * reverse[rp];
            }
        }
    }
}
