// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_column_density.hpp"
#include "ray_tracing.hpp"


// calc_column_density: calculate column density for given species for each cell and ray
// -------------------------------------------------------------------------------------

int calc_column_density (long ncells, CELL *cell, double *column, int spec)
{

  // For all cells n

# pragma omp parallel            \
  shared( cell, column, spec )   \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets

  for (long n = start; n < stop; n++)
  {


#   if (!CELL_BASED)

      long key[NCELLS];         // stores nrs. of cells on rays in order
      long raytot[NRAYS];       // cumulative nr. of evaluation points along each ray
      long cum_raytot[NRAYS];   // cumulative nr. of evaluation points along each ray

      EVALPOINT evalpoint[NCELLS];

      find_evalpoints (cell, evalpoint, key, raytot, cum_raytot, n);

      for (long r = 0; r < NRAYS; r++)
      {
        column[RINDEX(n,r)] = column_density (NCELLS, cell, evalpoint, key,
                                              raytot, cum_raytot, n, spec, r);
      }

#   else

      for (long r = 0; r < NRAYS; r++){

        column[RINDEX(n,r)] = cell_column_density (NCELLS, cell, n, spec, r);
      }

#   endif


  } /* end of n loop over grid points */
  } /* end of OpenMP parallel region */


  return(0);

}




// calc_column_densities: calculates column densities for species needed in chemistry
// ----------------------------------------------------------------------------------

int calc_column_densities (long ncells, CELL *cell, double *column_H2, double *column_HD,
                           double *column_C, double *column_CO)
{


  // For all cells n and rays r

# pragma omp parallel                                                                    \
  shared( cell, column_H2, column_HD, column_C, column_CO, H2_nr, HD_nr, C_nr, CO_nr )   \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;  // Note brackets


  for (long n = start; n < stop; n++)
  {


#   if (!CELL_BASED)

      long key[NCELLS];         // stores nrs. of cells on rays in order
      long raytot[NRAYS];       // cumulative nr. of evaluation points along each ray
      long cum_raytot[NRAYS];   // cumulative nr. of evaluation points along each ray

      EVALPOINT evalpoint[NCELLS];


      find_evalpoints(cell, evalpoint, key, raytot, cum_raytot, n);


      for (long r = 0; r < NRAYS; r++)
      {
        column_H2[RINDEX(n,r)] = column_density (NCELLS, cell, evalpoint, key, raytot,
                                                 cum_raytot, n, H2_nr, r);
        column_HD[RINDEX(n,r)] = column_density (NCELLS, cell, evalpoint, key, raytot,
                                                 cum_raytot, n, HD_nr, r);
        column_C[RINDEX(n,r)]  = column_density (NCELLS, cell, evalpoint, key, raytot,
                                                 cum_raytot, n, C_nr,  r);
        column_CO[RINDEX(n,r)] = column_density (NCELLS, cell, evalpoint, key, raytot,
                                                 cum_raytot, n, CO_nr, r);
      }

#   else

    for (long r = 0; r < NRAYS; r++)
    {
      column_H2[RINDEX(n,r)] = cell_column_density (NCELLS, cell, n, H2_nr, r);
      column_HD[RINDEX(n,r)] = cell_column_density (NCELLS, cell, n, HD_nr, r);
      column_C[RINDEX(n,r)]  = cell_column_density (NCELLS, cell, n, C_nr,  r);
      column_CO[RINDEX(n,r)] = cell_column_density (NCELLS, cell, n, CO_nr, r);
    }

#   endif


  } /* end of n loop over grid points */
  } /* end of OpenMP parallel region */


  return(0);

}




# if (!CELL_BASED)




// column_density: calculates column density for a species along a ray at a point
// ---------------------------------------------------------------------------------------

double column_density (long ncells, CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot,
                       long *cum_raytot, long gridp, int spec, long ray)
{

  double column_density_res = 0.0;   // resulting column density

  long etot = raytot[ray];


  if (etot > 0)
  {
    long evnr       = LOCAL_GP_NR_OF_EVALP(ray,0);
    long gridp_evnr = evnr;

    column_density_res = evalpoint[gridp_evnr].dZ * PC
                         *(cell[gridp].density*species[spec].abn[gridp]
                           + cell[evnr].density*species[spec].abn[evnr]) / 2.0;


    // Numerical integration along ray (line of sight)

    for (long e = 1; e < etot; e++)
    {
      long evnr       = LOCAL_GP_NR_OF_EVALP(ray,e);
      long evnrp      = LOCAL_GP_NR_OF_EVALP(ray,e-1);
      long gridp_evnr = evnr;

      column_density_res = column_density_res
                           + evalpoint[gridp_evnr].dZ * PC
                             * (cell[evnrp].density*species[spec].abn[evnrp]
                                + cell[evnr].density*species[spec].abn[evnr]) / 2.0;
    }

  }


  return column_density_res;

}




# else




// cell_column_density: calculates column density for a species along a ray at a point
// -----------------------------------------------------------------------------------

double cell_column_density (long ncells, CELL *cell, long origin, int spec, long ray)
{

  double column_density_res = 0.0;   // resulting column density


  // Walk along ray
  {
    double Z   = 0.0;
    double dZ  = 0.0;

    long current = origin;
    long next    = next_cell (NCELLS, cell, origin, ray, &Z, current, &dZ);


    while (next != NCELLS)
    {
      column_density_res = column_density_res
                           + dZ * PC * (cell[next].density*species[spec].abn[next]
                                        + cell[current].density*species[spec].abn[current]) / 2.0;

      current = next;
      next    = next_cell (NCELLS, cell, origin, ray, &Z, current, &dZ);
    }
  }


  return column_density_res;

}


#endif
