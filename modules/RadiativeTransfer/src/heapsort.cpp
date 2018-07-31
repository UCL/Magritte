// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <assert.h>
#include <vector>
using namespace std;

#include "heapsort.hpp"


long max (vector<double>& a, long n, long i, long j, long k)
{

  long m = i;


  if ( (j < n) && (a[j] > a[m]) )
  {
    m = j;
  }

  if ( (k < n) && (a[k] > a[m]) )
  {
    m = k;
  }


  return m;

}




int downheap (vector<double>& a, vector<long>& b, long n, long i)
{

  while (1)
  {
    long j = max (a, n, i, 2*i+1, 2*i+2);

    if (j == i)
    {
      break;
    }

    double temp1 = a[i];
    long   temp2 = b[i];

    a[i] = a[j];
    a[j] = temp1;

    b[i] = b[j];
    b[j] = temp2;

    i = j;
  }


  return (0);

}




int heapsort (vector<double>& a, vector<long>& b)
{

  // Get vector length

  const long n = a.size();


  // Assert that both vectors are equally long

  assert (n == b.size());


  // Sort vectors

  for (long i = (n-2)/2; i >=0 ; i--)
  {
    downheap (a, b, n, i);
  }

  for (long i = 0; i < n; i++)
  {
    double temp1 = a[n-i-1];
    long   temp2 = b[n-i-1];

    a[n-i-1] = a[0];
    a[0]     = temp1;

    b[n-i-1] = b[0];
    b[0]     = temp2;

    downheap (a, b, n-i-1, 0);
  }


  return (0);

}
