#include <cuda_runtime.h>


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
