// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "lines.hpp"



// source: calculate line source function
// --------------------------------------

int LINES::source (long ncells, CELLS *cells, int ls, double *source)
{


  for (int kr = 0; kr < nrad[ls]; kr++)
  {
    int i = irad[LSPECRAD(ls,kr)];   // i index corresponding to transition kr
    int j = jrad[LSPECRAD(ls,kr)];   // j index corresponding to transition kr

    long b_ij = LSPECLEVLEV(ls,i,j);   // A_coeff, B_coeff and frequency index
    long b_ji = LSPECLEVLEV(ls,j,i);   // A_coeff, B_coeff and frequency index

    double A_ij = A_coeff[b_ij];
    double B_ij = B_coeff[b_ij];
    double B_ji = B_coeff[b_ji];


#   pragma omp parallel                                                                              \
    shared (ncells, cells, A_ij, B_ij, B_ji, ls, kr, i, j, nrad, cum_nrad, nlev, cum_nlev, source)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {
      long s_ij = LSPECGRIDRAD(ls,p,kr);   // source and opacity index

      long p_i  = LINDEX(p,LSPECLEV(ls,i));    // pop index i
      long p_j  = LINDEX(p,LSPECLEV(ls,j));    // pop index j


      if ( (cells->pop[p_j] > POP_LOWER_LIMIT) || (cells->pop[p_i] > POP_LOWER_LIMIT) )
      {
        source[s_ij] = A_ij * cells->pop[p_i]  / (cells->pop[p_j]*B_ji - cells->pop[p_i]*B_ij);
      }

      else
      {
        source[s_ij] = 0.0;
      }


    } // end of o loop over grid points
    } // end of OpenMP parallel region

  } // end of kr loop over transitions


  return(0);

}




// opacity: calculate line opacity
// -------------------------------

int LINES::opacity (long ncells, CELLS *cells, int ls, double *opacity)
{

  for (int kr = 0; kr < nrad[ls]; kr++)
  {

    int i = irad[LSPECRAD(ls,kr)];   // i index corresponding to transition kr
    int j = jrad[LSPECRAD(ls,kr)];   // j index corresponding to transition kr

    long b_ij = LSPECLEVLEV(ls,i,j);   // A_coeff, B_coeff and frequency index
    long b_ji = LSPECLEVLEV(ls,j,i);   // A_coeff, B_coeff and frequency index

    double B_ij = B_coeff[b_ij];
    double B_ji = B_coeff[b_ji];


#   pragma omp parallel                                      \
    shared (ncells, cells, b_ij, B_ij, B_ji, ls, kr, i, j,   \
            nrad, cum_nrad, nlev, cum_nlev, opacity)         \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {
      long s_ij = LSPECGRIDRAD(ls,p,kr);   // source and opacity index

      long p_i  = LINDEX(p,LSPECLEV(ls,i));    // pop index i
      long p_j  = LINDEX(p,LSPECLEV(ls,j));    // pop index j

      double hv_4pi = HH * frequency[b_ij] / 4.0 / PI;


      opacity[s_ij] =  hv_4pi * (cells->pop[p_j]*B_ji - cells->pop[p_i]*B_ij);


      if (opacity[s_ij] < 1.0E-99)
      {
        opacity[s_ij] = 1.0E-99;
      }


    } // end of o loop over grid points
    } // end of OpenMP parallel region

  } // end of kr loop over transitions


  return(0);

}




// line_profile: calculate line profile function
// ---------------------------------------------

double profile (long ncells, CELLS *cells, double velocity, double freq, double line_freq, long o)
{

  double shift = line_freq * velocity / CC;
  double width = line_freq / CC * sqrt(2.0*KB*cells->temperature_gas[o]/MP + V_TURB*V_TURB);


  return exp( -pow((freq - line_freq - shift)/width, 2) ) / sqrt(PI) / width;

}
