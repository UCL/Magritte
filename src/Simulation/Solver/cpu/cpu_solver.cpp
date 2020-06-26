#include "cpu_solver.hpp"
#include "Simulation/simulation.hpp"


///  Constructor for cpuSolver
///    @param[in] ncells    : number of cells
///    @param[in] nfreqs    : number of frequency bins
///    @param[in] nlines    : number of lines
///    @param[in] nraypairs : number of ray pairs
///    @param[in] depth     : number of points along the ray pairs
//////////////////////////////////////////////////////////////////

cpuSolver :: cpuSolver (
    const Size ncells,
    const Size nfreqs,
    const Size nlines,
    const Size nboundary,
    const Size nraypairs,
    const Size depth,
    const Size n_off_diag )
    : Solver (ncells, nfreqs, nfreqs, nlines, nboundary, nraypairs, depth, n_off_diag)
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

    frequencies = new double[ncells*nfreqs_red];

    boundary_condition   = new BoundaryCondition[nboundary];
    boundary_temperature = new double           [nboundary];

    term1              = new double[area];
    term2              = new double[area];

    eta                = new double[area];
    chi                = new double[area];

    A                  = new double[area];
    a                  = new double[area];
    C                  = new double[area];
    c                  = new double[area];
    F                  = new double[area];
    G                  = new double[area];

    inverse_A          = new double[area];
    inverse_C          = new double[area];
    inverse_one_plus_F = new double[area];
    inverse_one_plus_G = new double[area];
     G_over_one_plus_G = new double[area];

    Su                 = new double[area];
    Sv                 = new double[area];
    dtau               = new double[area];

    L_diag             = new double[area];

    if (n_off_diag > 0)
    {
        L_upper = new double[n_off_diag*area];
        L_lower = new double[n_off_diag*area];
    }
}


///  Destructor for cpuSolver
/////////////////////////////

cpuSolver :: ~cpuSolver ()
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

void cpuSolver :: copy_model_data (const Model &model)
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
            frequencies[V(p,f)] = model.radiation.frequencies.nu[p][f];
        }
    }

}




void cpuSolver :: solve (
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
//        check_L         (w);
    }

    /// Store the result back in the model
    store (model);
}
