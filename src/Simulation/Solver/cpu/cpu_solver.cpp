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
    const Size nraypairs,
    const Size depth     )
    : Solver (ncells, nfreqs, nlines, nraypairs, depth)
{
    n1 = new Size[nraypairs_max];
    n2 = new Size[nraypairs_max];

    origins = new Size[nraypairs_max];
    reverse = new Real[nraypairs_max];

    nrs    = new Size[depth_max*nraypairs_max];
    shifts = new Real[depth_max*nraypairs_max];
    dZs    = new Real[depth_max*nraypairs_max];

    line            = new double[       nlines];
    line_emissivity = new double[ncells*nlines];
    line_opacity    = new double[ncells*nlines];
    line_width      = new double[ncells*nlines];

    frequencies     = new double[ncells*nfreqs];

    const size_t area = 10*depth_max*width_max;

    term1              = new Real[area];
    term2              = new Real[area];

    eta                = new Real[area];
    chi                = new Real[area];

    A                  = new Real[area];
    a                  = new Real[area];
    C                  = new Real[area];
    c                  = new Real[area];
    F                  = new Real[area];
    G                  = new Real[area];

    inverse_A          = new Real[area];
    inverse_C          = new Real[area];
    inverse_one_plus_F = new Real[area];
    inverse_one_plus_G = new Real[area];
     G_over_one_plus_G = new Real[area];

    Su                 = new Real[area];
    Sv                 = new Real[area];
    dtau               = new Real[area];

    L_diag             = new Real[area];
}


///  Destructor for cpuSolver
/////////////////////////////

cpuSolver :: ~cpuSolver ()
{
    delete[] n1;
    delete[] n2;

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
    }

    /// Store the result back in the model
    store (model);
}
