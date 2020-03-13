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
  }
};


///// Vector operations on CUDA's double3 vectors
/////////////////////////////////////////////////
//
//inline double3 operator+ (const double3 &a, const double3 &b)
//{
//  return make_double3 (a.x+b.x, a.y+b.y, a.z+b.z);
//}
//
//inline double3 operator- (const double3 &a, const double3 &b)
//{
//  return make_double3 (a.x-b.x, a.y-b.y, a.z-b.z);
//}
//
//inline double dot (const double3 &a, const double3 &b)
//{
//  // return a.x*b.x + a.y*b.y + a.z*b.z;
//  return fma(a.x, b.x, fma(a.y, b.y, a.z*b.z));
//}
//
//inline double infinity (void)
//{
//	const unsigned long long ieee754inf =  0x7ff0000000000000;
//
//	return 1.0E+250;//__longlong_as_double (ieee754inf);
//}

#endif // __MYCUDATOOLS_CUH_INCLUDED__
