#ifndef MAGRITTE_SOLVER_HPP
#define MAGRITTE_SOLVER_HPP


#include "Model/model.hpp"
#include "Tools/timer.hpp"
#include "queue.hpp"


#ifdef __CUDACC__
#   define HOST_DEVICE __host__ __device__
#   define HOST        __host__
#   define DEVICE      __device__
#   define GLOBAL      __global__
#else
#   define HOST_DEVICE
#   define HOST
#   define DEVICE
#   define GLOBAL
#endif


class DepthFirstDataLayout
{
    const Size depth_max;   ///< maximum depth of the ray block (lengths of the ray pairs)
    const Size width_max;   ///< maximum width of the ray block (nraypairs_max * nfreqs)

public:
    DepthFirstDataLayout (const Size depth_max, const Size width_max)
     : depth_max (depth_max)
     , width_max (width_max) {};

protected:
    HOST_DEVICE
    inline Size I (const Size i, const Size w) const {return i + w*depth_max;};
};


class WidthFirstDataLayout
{
    const Size depth_max;   ///< maximum depth of the ray block (lengths of the ray pairs)
    const Size width_max;   ///< maximum width of the ray block (nraypairs_max * nfreqs)

public:
    WidthFirstDataLayout (const Size depth_max, const Size width_max)
     : depth_max (depth_max)
     , width_max (width_max) {};

protected:
    HOST_DEVICE
    inline Size I (const Size i, const Size w) const {return w + i*width_max;};
};




template <typename Real, typename DataLayout>
struct Solver : private DataLayout
{
    const Size ncells;           ///< total number of cells
    const Size nfreqs;           ///< total number of frequency bins
    const Size nfreqs_red;       ///< total number of reduced frequency bins
    const Size nlines;           ///< total number of lines
    const Size nboundary;        ///< total number of cells on the boundary

    const Size nraypairs_max;    ///< maximum number or ray pairs in the ray block
          Size nraypairs;        ///< number or ray pairs in the ray block

    const Size depth_max;        ///< maximum depth of the ray block (lengths of the ray pairs)
    const Size width_max;        ///< maximum width of the ray block (nraypairs_max * nfreqs)
          Size width;            ///< width of the ray block (nraypairs * nfreqs)

    const Size area;             ///< depth_max * width_max

    const Size n_off_diag;       ///< number of off-diagonal rows on one side in L operator


    /// Constructor
    Solver (
        const Size ncells,
        const Size nfreqs,
        const Size nfreqs_red,
        const Size nlines,
        const Size nboundary,
        const Size nraypairs,
        const Size depth,
        const Size n_off_diag )
    : DataLayout    (depth, nraypairs * nfreqs_red)
    , ncells        (ncells)
    , nfreqs        (nfreqs)
    , nfreqs_red    (nfreqs_red)
    , nlines        (nlines)
    , nboundary     (nboundary)
    , nraypairs_max (nraypairs)
    , nraypairs     (nraypairs)
    , depth_max     (depth)
    , width_max     (nraypairs * nfreqs_red)
    , width         (nraypairs * nfreqs_red)
    , area          (depth_max * width_max)
    , n_off_diag    (n_off_diag) {};

    /// Copier for model data to solver (virtual)
    virtual void copy_model_data (const Model &model) = 0;

    /// Solve
    virtual void solve (
        const ProtoBlock &prb,
        const Size        R,
        const Size        r,
              Model      &model ) = 0;


    Real inverse_dtau_max = 100;   ///< inverse of maximal allowed optical depth increment

    Size RR;                       ///< absolute index of the ray direction
    Size rr;                       ///< relative index of the ray direction

    long first;                    ///< index of the first used element
    long last;                     ///< index of the last used element

    Size *n1;                      ///< number of encountered cells in ray 1
    Size *n2;                      ///< number of encountered cells in ray 2

    Size *n_tot;                   ///< number of cells on raypair (n1 + n2 + 1)

    Size n1_min;                   ///< minimum of n1

    Size   *origins;               ///< origins of the ray pairs in the ray block
    double *reverse;               ///< 1.0 if ray_ar is longer than ray_r (-1.0 otherwise)

    Size *bdy_0;                   ///< boundary index of first point on the ray
    Size *bdy_n;                   ///< boundary index of last point on the ray


    /// Pointers to model data
    double *line;
    double *line_emissivity;
    double *line_opacity;
    double *line_width;

    Real   *frequencies;

    BoundaryCondition *boundary_condition;
    double            *boundary_temperature;


    Size   *nrs;                ///< cell nr corresp to point along ray
    double *shifts;             ///< Doppler shift scale factors along ray
    double *dZs;                ///< distance increments corresp to point along ray

    Real *term1;                ///< effective source for u along the ray
    Real *term2;                ///< effective source for v along the ray

    Real *eta;                  ///< emissivities
    Real *chi;                  ///< opacities

    Real *A;                    ///< A coefficient in Feautrier recursion relation
    Real *a;                    ///< a coefficient in Feautrier recursion relation
    Real *C;                    ///< C coefficient in Feautrier recursion relation
    Real *c;                    ///< c coefficient in Feautrier recursion relation
    Real *F;                    ///< helper variable from Rybicki & Hummer (1991)
    Real *G;                    ///< helper variable from Rybicki & Hummer (1991)

    Real *inverse_A;            ///< helper variable
    Real *inverse_C;            ///< helper variable
    Real *inverse_one_plus_F;   ///< helper variable
    Real *inverse_one_plus_G;   ///< helper variable
    Real * G_over_one_plus_G;   ///< helper variable

    Real *Su;                   ///< effective source for u along the ray
    Real *Sv;                   ///< effective source for v along the ray
    Real *dtau;                 ///< optical depth increment along the ray

    Real *L_upper;              ///< upper-half of L matrix
    Real *L_diag;               ///< diagonal of L matrix
    Real *L_lower;              ///< lower-half of L matrix


    /// Indices
//    HOST_DEVICE
//    inline Size I (const Size i, const Size w) const {return i + w*depth_max;};
//    inline Size I (const Size i, const Size w) const {return w + i*width;     };

//    virtual HOST_DEVICE
//    Size I (const Size i, const Size w) const = 0;
// Cannot pass a class with virtual methods to a kernel, see
// https://stackoverflow.com/questions/12701170/cuda-virtual-class
// Following Mark Harris' solution of policy classes
// https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design

    using DataLayout::I;

    HOST_DEVICE
    inline Size D (const Size i, const Size d) const {return d + i*depth_max; };
    HOST_DEVICE
    inline Size L (const Size i, const Size l) const {return l + i*nlines;    };
    HOST_DEVICE
    inline Size V (const Size i, const Size f) const {return f + i*nfreqs_red;};
    HOST_DEVICE
    inline Size M (const Size m, const Size i) const {return i + m*area;      };


    void setup (
        const Model      &model,
        const Size        R,
        const Size        r,
        const ProtoBlock &prb   );


    inline void add_L_diag (
        const Thermodynamics &thermodyn,
        const double          invr_mass,
        const double          freq_line,
        const double          constante,
        const Size            rp,
        const Size            f,
        const Size            k,
              Lambda         &lambda     ) const;

    inline void add_L_lower (
        const Thermodynamics &thermodyn,
        const double          invr_mass,
        const double          freq_line,
        const double          constante,
        const Size            rp,
        const Size            f,
        const Size            k,
        const Size            m,
              Lambda         &lambda     ) ;//const;

    inline void add_L_upper (
        const Thermodynamics &thermodyn,
        const double          invr_mass,
        const double          freq_line,
        const double          constante,
        const Size            rp,
        const Size            f,
        const Size            k,
        const Size            m,
              Lambda         &lambda     ) ;//const;

    inline void update_Lambda (Model &model) ;//const;


    inline void store (Model &model) ;//const;

    HOST_DEVICE
    inline void get_eta_and_chi (
        const Size  Dn,
        const Real  frequency,
              Real &eta,
              Real &chi         );

    HOST_DEVICE
    inline void get_eta_and_chi (
        const Size In,
        const Size Dn,
        const Real frequency    );

    HOST_DEVICE
    inline void solve_Feautrier (const Size w)
    {
        solve_2nd_order_Feautrier_non_adaptive(w);
//        solve_2nd_order_Feautrier_adaptive(w);
//        solve_4th_order_Feautrier_non_adaptive(w);
//        solve_4th_order_Feautrier_adaptive(w);
    }

    HOST_DEVICE
    inline void solve_2nd_order_Feautrier_non_adaptive (const Size w);
    HOST_DEVICE
    inline void solve_2nd_order_Feautrier_adaptive     (const Size w);
    HOST_DEVICE
    inline void solve_4th_order_Feautrier_non_adaptive (const Size w);
    HOST_DEVICE
    inline void solve_4th_order_Feautrier_adaptive     (const Size w);

    // Boundary conditions
    inline Real boundary_intensity (const Size bdy_nr, const Real frequency) const;


    // Helper variables for tests
    MatrixXd test_T;   ///< tridiagonal T matrix
    MatrixXd test_L;   ///< matrix representation of L operator
    VectorXd test_s;   ///< effective source function
    VectorXd test_u;   ///< solution u

    inline void setup_T (const Size w);
    inline void setup_L (const Size w);

    inline Real check_L (const Size w);


    singleTimer timer;


private:

    HOST_DEVICE
    inline Real my_fma (const Real a, const Real b, const Real c) const;
    HOST_DEVICE
    inline Real gaussian (const Real width, const Real diff) const;
    HOST_DEVICE
    inline Real planck (const Real temperature, const Real frequency) const;

    const Real one = 1.0;

};


#include "solver.tpp"
#include "tests.tpp"


#endif //MAGRITTE_SOLVER_HPP
