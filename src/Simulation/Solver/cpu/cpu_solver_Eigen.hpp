// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________

#pragma once

#include "Simulation/Solver/solver.hpp"
#include "Model/model.hpp"
#include "Tools/timer.hpp"
#include "Tools/types.hpp"



#ifdef USETIMERS

#   define TIMER_TIC(tic)                       \
         clock_t tic;                           \
         if (threadIdx.x == 0) tic = clock64();
#   define TIMER_TOC(tic,name)                  \
         if (threadIdx.x == 0) printf("%s : %ld\n", name, clock64()-tic);
#   define PRINTLINE                            \
         if (threadIdx.x == 0) printf("-----------------------------\n");

#else

#   define TIMER_TIC(tic)
#   define TIMER_TOC(tic, name)
#   define PRINTLINE

#endif

// if (threadIdx.x == 0) atomicAdd(&time, (toc>tic) ? (toc-tic) : (toc+(0xffffffff-tic))); \

#define HANDLE_ERROR(body)  \
    cudaError_t err = body; \
    if (err!=cudaSuccess) printf ("CUDA ERROR : %s", cudaGetErrorString(err));




///  Raypair: data structure for a pair of rays
///////////////////////////////////////////////

struct cpuSolverEigen : public Solver<VectorXd>
{

    Size gpuBlockSize = 32;
    Size gpuNumBlocks = 32;


    /// Constructor
    cpuSolverEigen (
        const Size ncells,
        const Size nfreqs,
        const Size nlines,
        const Size nboundary,
        const Size nraypairs,
        const Size depth,
        const Size n_off_diag);

    // Destructor
    ~cpuSolverEigen();


    void copy_model_data (const Model &model) override;

    void solve (
        const ProtoBlock &prb,
        const Size        R,
        const Size        r,
              Model      &model) override;


    inline void store (Model &model) const;


    // Invert the data layout. DOES NOT WORK, DOES NOT OVERRIDE BASE CLASS FUNCTION!!!
//    HOST_DEVICE
//    Size I (const Size i, const Size w) const {return i + w*depth_max;};

};
