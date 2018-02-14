// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "declarations.hpp"
#include "calc_UV_field.hpp"


// calc_UV_field: calculates UV radiation field at each cell
// ---------------------------------------------------------

int calc_UV_field (long ncells, CELL *cell)
{

  const double tau_UV = 3.02;   // conversion factor from visual extinction to UV attenuation


  // For all grid points

# pragma omp parallel              \
  shared (ncells, cell, antipod)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    cell[n].UV = 0.0;


    // External UV radiation field
    // _ _ _ _ _ _ _ _ _ _ _ _ _ _


    // For all rays

    for (long r = 0; r < NRAYS; r++)
    {
      cell[n].UV = cell[n].UV + cell[n].ray[r].rad_surface*exp(-tau_UV*cell[n].ray[r].AV);
    }




    // Internal UV radiation field
    // _ _ _ _ _ _ _ _ _ _ _ _ _ _


  } // end of n loop over grid points
  } // end of OpenMP parallel region


  return(0);

}
