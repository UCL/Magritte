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


typedef double Real;
typedef size_t Size;

//template <typename Real>
struct Solver
{
    const size_t ncells;           ///< total number of cells
    const size_t nfreqs;           ///< total number of frequency bins
    const size_t nlines;           ///< total number of lines

    const size_t nraypairs_max;    ///< maximum number or ray pairs in the ray block
          size_t nraypairs;        ///< number or ray pairs in the ray block

    const size_t depth_max;        ///< maximum depth of the ray block (lengths of the ray pairs)
    const size_t width_max;        ///< maximum width of the ray block (nraypairs_max * nfreqs)
          size_t width;            ///< width of the ray block (nraypairs * nfreqs)



    /// Constructor
    Solver (
        const size_t ncells,
        const size_t nfreqs,
        const size_t nlines,
        const size_t nraypairs,
        const size_t depth     )
    : ncells        (ncells)
    , nfreqs        (nfreqs)
    , nlines        (nlines)
    , nraypairs_max (nraypairs)
    , nraypairs     (nraypairs)
    , depth_max     (depth)
    , width_max     (nraypairs * nfreqs)
    , width         (nraypairs * nfreqs) {};

    /// Copier for model data to solver (virtual)
    virtual void copy_model_data (const Model &model) = 0;

    /// Solve
    virtual void solve (
        const ProtoBlock &prb,
        const size_t      R,
        const size_t      r,
              Model      &model ) = 0;


    Real   inverse_dtau_max = 1.0;   ///< inverse of the maximal allowed optical depth increment
    size_t n_off_diag       = 0;     ///< number of off-diagonal rows on one side (default = 0)

    size_t RR;                       ///< absolute index of the ray direction
    size_t rr;                       ///< relative index of the ray direction

    size_t first;                    ///< index of the first used element
    size_t last;                     ///< index of the last used element

    size_t *n1;                      ///< number of encountered cells in ray 1
    size_t *n2;                      ///< number of encountered cells in ray 2

    size_t n1_min;                   ///< minimum of n1

    size_t *origins;                 ///< origins of the ray pairs in the ray block
    double *reverse;                 ///< 1.0 if ray_ar is longer than ray_r (-1.0 otherwise)



    /// Pointers to model data
    double *line;
    double *line_emissivity;
    double *line_opacity;
    double *line_width;

    double *frequencies;

    size_t *nrs;                  ///< cell number corresponding to point along the ray
    double *shifts;               ///< Doppler shift scale factors along the ray
    double *dZs;                  ///< distance increments corresponding to point along the ray

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

    // vReal2 L_upper;   ///< upper-half of L matrix
    Real *L_diag;    ///< diagonal   of L matrix
    // vReal2 L_lower;   ///< lower-half of L matrix

    /// Indices
    inline size_t I (const size_t i, const size_t w) const {return w + i*width;    };
    inline size_t D (const size_t i, const size_t d) const {return d + i*depth_max;};
    inline size_t L (const size_t i, const size_t l) const {return l + i*nlines;   };
    inline size_t V (const size_t i, const size_t f) const {return f + i*nfreqs;   };


    void setup (
        const Model      &model,
        const size_t      R,
        const size_t      r,
        const ProtoBlock &prb   );

    void store (Model &model) const;

    HOST_DEVICE
    void get_eta_and_chi (
        const size_t Dn,
        const Real   frequency,
              Real  &eta,
              Real  &chi       );

    HOST_DEVICE
    void solve_Feautrier (const size_t w);

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

};


#include "solver.tpp"


#endif //MAGRITTE_SOLVER_HPP
