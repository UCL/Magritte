#ifndef __MYCUDATOOLS_CUH_INCLUDED__
#define __MYCUDATOOLS_CUH_INCLUDED__


#include <cuda_runtime.h>

// #define USETIMERS


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


///  A base structure for using Unified memory
//////////////////////////////////////////////

struct Managed
{
  /// Overloading the new operator
  void *operator new (size_t len)
  {
    void *ptr;
    cudaMallocManaged (&ptr, len);
    cudaDeviceSynchronize ();
    return ptr;
  }

  /// Overloading the delete operator
  void operator delete (void *ptr)
  {
    cudaDeviceSynchronize ();
    cudaFree (ptr);
    return;
  }
};


#endif // __MYCUDATOOLS_CUH_INCLUDED__
