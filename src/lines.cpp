// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "lines.hpp"



// line_source: calculate line source function
// -------------------------------------------

int line_source (int *irad, int *jrad, double *A_coeff, double *B_coeff, double *pop, int lspec,
                 double *source)
{


  for (int kr = 0; kr < nrad[lspec]; kr++)
  {
    int i = irad[LSPECRAD(lspec,kr)];   // i index corresponding to transition kr
    int j = jrad[LSPECRAD(lspec,kr)];   // j index corresponding to transition kr

    long b_ij = LSPECLEVLEV(lspec,i,j);   // A_coeff, B_coeff and frequency index
    long b_ji = LSPECLEVLEV(lspec,j,i);   // A_coeff, B_coeff and frequency index

    double A_ij = A_coeff[b_ij];
    double B_ij = B_coeff[b_ij];
    double B_ji = B_coeff[b_ji];


#   pragma omp parallel                                                                        \
    shared( A_ij, B_ij, B_ji, pop, lspec, kr, i, j, nrad, cum_nrad, nlev, cum_nlev, source )   \
    default( none )
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long gridp = start; gridp < stop; gridp++)
    {
      long s_ij = LSPECGRIDRAD(lspec,gridp,kr);   // source and opacity index

      long p_i  = LSPECGRIDLEV(lspec,gridp,i);    // pop index i
      long p_j  = LSPECGRIDLEV(lspec,gridp,j);    // pop index j


      if ( (pop[p_j] > POP_LOWER_LIMIT) || (pop[p_i] > POP_LOWER_LIMIT) )
      {
        source[s_ij]  = A_ij * pop[p_i]  / (pop[p_j]*B_ji - pop[p_i]*B_ij);
      }

      else
      {
        source[s_ij]  = 0.0;
      }


    } // end of gridp loop over grid points
    } // end of OpenMP parallel region

  } // end of kr loop over transitions


  return(0);

}




// line_opacity: calculate line opacity
// ------------------------------------

int line_opacity (int *irad, int *jrad, double *frequency, double *B_coeff, double *pop, int lspec,
                  double *opacity)
{

  for (int kr=0; kr<nrad[lspec]; kr++){

    int i = irad[LSPECRAD(lspec,kr)];   // i index corresponding to transition kr
    int j = jrad[LSPECRAD(lspec,kr)];   // j index corresponding to transition kr

    long b_ij = LSPECLEVLEV(lspec,i,j);   // A_coeff, B_coeff and frequency index
    long b_ji = LSPECLEVLEV(lspec,j,i);   // A_coeff, B_coeff and frequency index

    double B_ij = B_coeff[b_ij];
    double B_ji = B_coeff[b_ji];


#   pragma omp parallel                                                                          \
    shared( frequency, b_ij, B_ij, B_ji, pop, lspec, kr, i, j, nrad, cum_nrad, nlev, cum_nlev,   \
            opacity )                                                                            \
    default( none )
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long gridp = start; gridp < stop; gridp++)
    {
      long s_ij = LSPECGRIDRAD(lspec,gridp,kr);   // source and opacity index

      long p_i  = LSPECGRIDLEV(lspec,gridp,i);    // pop index i
      long p_j  = LSPECGRIDLEV(lspec,gridp,j);    // pop index j

      double hv_4pi = HH * frequency[b_ij] / 4.0 / PI;


      opacity[s_ij] =  hv_4pi * (pop[p_j]*B_ji - pop[p_i]*B_ij);


      if (opacity[s_ij] < 1.0E-99)
      {
        opacity[s_ij] = 1.0E-99;
      }


    } // end of gridp loop over grid points
    } // end of OpenMP parallel region

  } // end of kr loop over transitions


  return(0);

}




#if (!CELL_BASED)




// line_profile: calculate line profile function
// ---------------------------------------------

double line_profile (long ncells, CELL *cell, EVALPOINT *evalpoint,
                     double freq, double line_freq, long gridp)
{

  double shift = line_freq * evalpoint[gridp].vol / CC;

  double width = line_freq / CC * sqrt(2.0*KB*cell[gridp].temperature.gas/MP + V_TURB*V_TURB);


  return exp( -pow((freq - line_freq - shift)/width, 2) ) / sqrt(PI) / width;

}




#else




// cell_line_profile: calculate line profile function
// --------------------------------------------------

double cell_line_profile (long ncells, CELL *cell, double velocity,
                          double freq, double line_freq, long gridp)
{

  double shift = line_freq * velocity / CC;

  double width = line_freq / CC * sqrt(2.0*KB*cell[gridp].temperature.gas/MP + V_TURB*V_TURB);


  return exp( -pow((freq - line_freq - shift)/width, 2) ) / sqrt(PI) / width;

}


#endif
