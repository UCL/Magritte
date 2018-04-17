// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "calc_column_density.hpp"
#include "ray_tracing.hpp"


// calc_column_density: calculate column density for given species for each cell and ray
// -------------------------------------------------------------------------------------

int calc_column_tot (CELLS *cells, RAYS rays)
{

  int spec = NSPEC-1;   // Last species has fractional abundance 1.0, so traces total density.


  // For all cells n

# pragma omp parallel          \
  shared (cells, rays, spec)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets

  for (long p = start; p < stop; p++)
  {
    for (long r = 0; r < NRAYS; r++)
    {
      cells->column_tot[RINDEX(p,r)] = column_density (cells, rays, p, spec, r);
    }
  } // end of n loop over grid points
  } // end of OpenMP parallel region


  return(0);

}




// calc_column_densities: calculates column densities for species needed in chemistry
// ----------------------------------------------------------------------------------

int calc_column_densities (CELLS *cells, RAYS rays, SPECIES species)
{


  // For all cells n and rays r

# pragma omp parallel             \
  shared (cells, rays, species)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;  // Note brackets


  for (long p = start; p < stop; p++)
  {
    for (long r = 0; r < NRAYS; r++)
    {
      cells->column_H2[RINDEX(p,r)] = column_density (cells, rays, p, species.nr_H2, r);
      cells->column_HD[RINDEX(p,r)] = column_density (cells, rays, p, species.nr_HD, r);
      cells->column_C[RINDEX(p,r)]  = column_density (cells, rays, p, species.nr_C,  r);
      cells->column_CO[RINDEX(p,r)] = column_density (cells, rays, p, species.nr_CO, r);
    }
  } // end of n loop over grid points
  } // end of OpenMP parallel region


  return(0);

}




// column_density: calculates column density for a species along a ray at a point
// ------------------------------------------------------------------------------

double column_density (CELLS *cells, RAYS rays, long origin, int spec, long ray)
{

  double column_res = 0.0;   // resulting column density


  // Walk along ray

  double Z   = 0.0;
  double dZ  = 0.0;

  long current = origin;
  long next    = next_cell (NCELLS, cells, rays, origin, ray, &Z, current, &dZ);


  while (next != NCELLS)
  {
    column_res = column_res
                 + dZ*PC*(cells->density[next]*cells->abundance[SINDEX(next,spec)]
                          + cells->density[current]*cells->abundance[SINDEX(current,spec)])
                         / 2.0;

    current = next;
    next    = next_cell (NCELLS, cells, rays, origin, ray, &Z, current, &dZ);
  }

  column_res = column_res + cells->column[RINDEX(current,ray)];


  return column_res;

}
