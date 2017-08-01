/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Heapsort algorithm for a list and its indices                                                 */
/*                                                                                               */
/* (based on a heapsort code from www.rosattacode.org)                                           */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*     a = array of doubles to sort   (IN/OUT)                                                   */
/*     b = array of identifiers       (IN/OUT)                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>

#include "heapsort.hpp"


long max (double *a, long n, long i, long j, long k)
{

  long m = i;


  if (j<n && a[j]>a[m]) {
    m = j;
  }

  if (k<n && a[k]>a[m]) {
    m = k;
  }

  return m;
}



void downheap(double *a, long *b, long n, long i)
{

  while (1){

    long j=max(a, n, i, 2*i+1, 2*i+2);

    if (j == i){
      break;
    }

    double temp1 = a[i];
    long    temp2 = b[i];

    a[i] = a[j];
    a[j] = temp1;

    b[i] = b[j];
    b[j] = temp2;

    i = j;

  }
}

void heapsort(double *a, long *b, long n)
{

  long i;

  for (i=(n-2)/2; i>=0; i--){

    downheap(a, b, n, i);
  }

  for (i=0; i<n; i++){

    double temp1 = a[n-i-1];
    long    temp2 = b[n-i-1];

    a[n-i-1] = a[0];
    a[0]     = temp1;

    b[n-i-1] = b[0];
    b[0]     = temp2;

    downheap(a, b, n-i-1, 0);
    }
}



/* This main is for testing the heapsort algorithm above                                         */
/*---------------------------------------------------------------------------------------------- */
/*
int main()
{
  double a[] = {3.14, 7, 5, 1.3, -2.1};
  long    b[] = {3, 1, 2, 5, 4};
  long n = sizeof a / sizeof a[0];
  long i;

  for (i=0; i < n; i++){

    printf("%3.1f ", a[i]);
  }

  printf("\n");

  for (i=0; i < n; i++){

    printf("%ld ", b[i]);
  }


  printf("\n");
  printf("\n");


  heapsort(a, b, n);


  for (i = 0; i < n; i++){

    printf("%3.1f ", a[i]);
  }

  printf("\n");

  for (i=0; i < n; i++){

    printf("%ld ", b[i]);
  }

  printf("\n");


    return 0;
}
*/
/*-----------------------------------------------------------------------------------------------*/
