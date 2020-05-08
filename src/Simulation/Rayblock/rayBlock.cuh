// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________

#pragma once

#include "Model/model.hpp"
#include "myCudaTools.cuh"
#include "Tools/timer.hpp"


///  Raypair: data structure for a pair of rays
///////////////////////////////////////////////

class CUDA_RayBlock : public Managed, RayBlock
{

public:

    Size gpuBlockSize = 32;
    Size gpuNumBlocks = 32;


    /// Constructor
    __host__ RayBlock (
        const Size ncells,
        const Size nfreqs,
        const Size nlines,
        const Size nraypairs,
        const Size depth     )

    /// Destructor
    __host__ ~RayBlock ();

};
