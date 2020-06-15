#ifndef MAGRITTE_SOLVER_HPP
#define MAGRITTE_SOLVER_HPP


#include "Model/model.hpp"
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


//typedef double Real;

template <typename Real>
struct Solver
{
    const Size ncells;           ///< total number of cells
    const Size nfreqs;           ///< total number of frequency bins
    const Size nfreqs_red;       ///< total number of reduced frequency bins
    const Size nlines;           ///< total number of lines

    const Size nraypairs_max;    ///< maximum number or ray pairs in the ray block
          Size nraypairs;        ///< number or ray pairs in the ray block

    const Size depth_max;        ///< maximum depth of the ray block (lengths of the ray pairs)
    const Size width_max;        ///< maximum width of the ray block (nraypairs_max * nfreqs)
          Size width;            ///< width of the ray block (nraypairs * nfreqs)

    const Size n_off_diag;       ///< number of off-diagonal rows on one side in L operator


    /// Constructor
    Solver (
        const Size ncells,
        const Size nfreqs,
        const Size nfreqs_red,
        const Size nlines,
        const Size nraypairs,
        const Size depth,
        const Size n_off_diag )
    : ncells        (ncells)
    , nfreqs        (nfreqs)
    , nfreqs_red    (nfreqs_red)
    , nlines        (nlines)
    , nraypairs_max (nraypairs)
    , nraypairs     (nraypairs)
    , depth_max     (depth)
    , width_max     (nraypairs * nfreqs_red)
    , width         (nraypairs * nfreqs_red)
    , n_off_diag    (n_off_diag) {};

    /// Copier for model data to solver (virtual)
    virtual void copy_model_data (const Model &model) = 0;

    /// Solve
    virtual void solve (
        const ProtoBlock &prb,
        const Size        R,
        const Size        r,
              Model      &model ) = 0;


    Real inverse_dtau_max = 1.0;   ///< inverse of maximal allowed optical depth increment

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



    /// Pointers to model data
    double *line;
    double *line_emissivity;
    double *line_opacity;
    double *line_width;

    Real   *frequencies;

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
    HOST_DEVICE
    inline Size I (const Size i, const Size w) const {return i + w*depth_max;};
//    inline Size I (const Size i, const Size w) const {return w + i*width;     };

//    virtual HOST_DEVICE
//    Size I (const Size i, const Size w) const = 0;

    HOST_DEVICE
    inline Size D (const Size i, const Size d) const {return d + i*depth_max; };
    HOST_DEVICE
    inline Size L (const Size i, const Size l) const {return l + i*nlines;    };
    HOST_DEVICE
    inline Size V (const Size i, const Size f) const {return f + i*nfreqs_red;};
    HOST_DEVICE
    inline Size M (const Size m, const Size i) const {return i + m*n_off_diag;};


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
              Lambda         &lambda     ) const;

    inline void add_L_upper (
        const Thermodynamics &thermodyn,
        const double          invr_mass,
        const double          freq_line,
        const double          constante,
        const Size            rp,
        const Size            f,
        const Size            k,
        const Size            m,
              Lambda         &lambda     ) const;

    inline void update_Lambda (Model &model) const;


    inline void store (Model &model) const;

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


    //    inline double get_L_diag (
    //        const Thermodynamics &thermodynamics,
    //        const double          inverse_mass,
    //        const double          freq_line,
    //        const int             lane           ) const;

    //    inline double get_L_lower (
    //        const Thermodynamics &thermodynamics,
    //        const double          inverse_mass,
    //        const double          freq_line,
    //        const int             lane,
    //        const long            m              ) const;

    //    inline double get_L_upper (
    //        const Thermodynamics &thermodynamics,
    //        const double          inverse_mass,
    //        const double          freq_line,
    //        const int             lane,
    //        const long            m              ) const;

    //    inline void update_Lambda (
    //        const Frequencies    &frequencies,
    //        const Thermodynamics &thermodynamics,
    //        const long            p,
    //        const long            f,
    //        const double          weight_angular,
    //              Lines          &lines          ) const;

private:

    HOST_DEVICE
    inline Real my_fma (const Real a, const Real b, const Real c) const;
    HOST_DEVICE
    inline Real gaussian (const Real width, const Real diff) const;
    HOST_DEVICE
    inline Real planck (const Real temperature, const Real frequency) const;

//    HOST_DEVICE
//    inline Real expf (const Real x) const;
//    HOST_DEVICE
//    inline Real expm1 (const Real x) const;

    const Real one = 1.0;

//    const double inverse_index[40] =
//        {    0.,     1., 1./ 2., 1./ 3., 1./ 4., 1./ 5., 1./ 6, 1./ 7, 1./ 8., 1./ 9.,
//         1./10., 1./11., 1./12., 1./13., 1./14., 1./15., 1./16, 1./17, 1./18., 1./19.,
//         1./20., 1./21., 1./22., 1./23., 1./24., 1./25., 1./26, 1./27, 1./28., 1./29.,
//         1./30., 1./31., 1./32., 1./33., 1./34., 1./35., 1./36, 1./37, 1./38., 1./39. };

};


#include "solver.tpp"


#endif //MAGRITTE_SOLVER_HPP
