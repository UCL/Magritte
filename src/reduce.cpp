// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "reduce.hpp"
#include "initializers.hpp"
#include "ray_tracing.hpp"


// reduce: reduce number of cells, return resulting number of cells
// ----------------------------------------------------------------

long reduce (CELLS *cells)
{

  // Specify grid boundaries

  double x_min = X_MIN;
  double x_max = X_MAX;
  double y_min = Y_MIN;
  double y_max = Y_MAX;
  double z_min = Z_MIN;
  double z_max = Z_MAX;

  double threshold = THRESHOLD;   // keep cells if rel_density_change > threshold


  // Crop grid

  crop (cells, x_min, x_max, y_min, y_max, z_min, z_max);


  // Reduce grid

  density_reduce (cells, threshold);


  // Set id's to relate grid and reduced grid, get ncells_red

  long ncells_red = set_ids (cells);


  return ncells_red;

}




// crop: crop spatial range of data
// --------------------------------

int crop (CELLS *cells,
          double x_min, double x_max,
          double y_min, double y_max,
          double z_min, double z_max)
{

  long ncells = cells->ncells;


# pragma omp parallel                                                \
  shared (ncells, cells, x_min, x_max, y_min, y_max, z_min, z_max)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets

  for (long p = start; p < stop; p++)
  {
    if (    (x_min > cells->x[p]) || (cells->x[p] > x_max)
         || (y_min > cells->y[p]) || (cells->y[p] > y_max)
         || (z_min > cells->z[p]) || (cells->z[p] > z_max) )
    {
      cells->removed[p] = true;
    }
  }
  } // end of OpenMP parallel region


  return (0);

}




// density_reduce: reduce number of cell in regions of constant density
// --------------------------------------------------------------------

int density_reduce (CELLS *cells, double min_density_change)
{

  long ncells = cells->ncells;


  // Note that this loop cannot be paralellized !

  for (long p = 0; p < ncells; p++)
  {
    if (!cells->removed[p])
    {

      double density_c  = cells->density[p];   // density of current cell

      cells->removed[p] = true;                // assume cell can be removed


      // Check whether cell can indeed be removed

      for (int n = 0; n < cells->n_neighbors[p]; n++)
      {
        long nr = cells->neighbor[RINDEX(p,n)];

        double rel_density_change = 2.0*fabs(cells->density[nr] - density_c)
                                          / (cells->density[nr] + density_c);


        // Do not remove if density changes too much or neighbor was removed
        // The latter to avoid large gaps being formed

        if ( (rel_density_change > min_density_change) || cells->removed[nr] || cells->boundary[nr] )
        {
          cells->removed[p] = false;
        }
      }

    }
  }


  return (0);

}




// set_ids: determine cell numbers in the reduced grid, return nr of reduced cells
// -------------------------------------------------------------------------------

long set_ids (CELLS *cells)
{

  long ncells = cells->ncells;

  long cell_id_reduced = 0;


  // Store nr of cell in reduced grid in id's of full grid
  // Note that this loop cannot be paralellized !

  for (long p = 0; p < ncells; p++)
  {
    if (cells->removed[p])
    {
      cells->id[p] = -1;
    }

    else
    {
      cells->id[p] = cell_id_reduced;

      cell_id_reduced++;
    }
  }


  // cell_id_reduced now equals total number of cells in reduced grid

  return cell_id_reduced;

}




// initialized_reduced_grid: initialize reduced grid
// -------------------------------------------------

int initialize_reduced_grid (CELLS *cells_red, CELLS *cells, RAYS rays)
{

  long ncells     = cells->ncells;
  long ncells_red = cells_red->ncells;


  cells_red->initialize ();


# pragma omp parallel                             \
  shared (ncells_red, cells_red, ncells, cells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    if (!cells->removed[p])
    {
      long nr = cells->id[p];   // nr of cell n in reduced grid

      cells_red->x[nr] = cells->x[p];
      cells_red->y[nr] = cells->y[p];
      cells_red->z[nr] = cells->z[p];

      cells_red->vx[nr] = cells->vx[p];
      cells_red->vy[nr] = cells->vy[p];
      cells_red->vz[nr] = cells->vz[p];

      cells_red->density[nr] = cells->density[p];

      cells_red->temperature_gas[nr]      = cells->temperature_gas[p];
      cells_red->temperature_dust[nr]     = cells->temperature_dust[p];
      cells_red->temperature_gas_prev[nr] = cells->temperature_gas_prev[p];

      cells_red->thermal_ratio[nr]      = cells->thermal_ratio[p];
      cells_red->thermal_ratio_prev[nr] = cells->thermal_ratio_prev[p];

      cells_red->UV[nr] = cells->UV[p];


      for (int s = 0; s < NSPEC; s++)
      {
        cells_red->abundance[SINDEX(nr,s)] = cells->abundance[SINDEX(p,s)];
      }

      for (int e = 0; e < NREAC; e++)
      {
        cells_red->rate[READEX(nr,e)] = cells->rate[READEX(p,e)];
      }

      for (long r = 0; r < NRAYS; r++)
      {
        cells_red->intensity[RINDEX(nr,r)]   = cells->intensity[RINDEX(p,r)];
        cells_red->column[RINDEX(nr,r)]      = cells->column[RINDEX(p,r)];
        cells_red->rad_surface[RINDEX(nr,r)] = cells->rad_surface[RINDEX(p,r)];
        cells_red->AV[RINDEX(nr,r)]          = cells->AV[RINDEX(p,r)];
      }

      for (int l = 0; l < TOT_NLEV; l++)
      {
        cells_red->pop[LINDEX(nr,l)] = cells->pop[LINDEX(p,l)];
      }

      for (int k = 0; k < TOT_NRAD; k++)
      {
        cells_red->mean_intensity[KINDEX(nr,k)] = cells->mean_intensity[KINDEX(p,k)];
      }

    }
  }
  } // end of OpenMP parallel region


  // Find neighboring cells for each cell

  find_neighbors (ncells_red, cells_red, rays);


  // Find endpoint of each ray for each cell

  find_endpoints (ncells_red, cells_red, rays);


  return (0);
}




// interpolate: interpolate reduced grid back to original grid
// -----------------------------------------------------------

int interpolate (CELLS *cells_red, CELLS *cells)
{

  long ncells     = cells->ncells;
  long ncells_red = cells_red->ncells;


# pragma omp parallel                             \
  shared (ncells_red, cells_red, ncells, cells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    if (cells->removed[p])
    {

      // Take average of neighbors

      cells->density[p]              = 0.0;
      cells->temperature_gas[p]      = 0.0;
      cells->temperature_gas_prev[p] = 0.0;
      cells->thermal_ratio[p]        = 0.0;
      cells->thermal_ratio_prev[p]   = 0.0;


      for (int n = 0; n < cells->n_neighbors[p]; n++)
      {
        long nr = cells->neighbor[RINDEX(p,n)];   // nr of neighbor in grid

        if (!cells->removed[nr])
        {
          long nr_red = cells->id[nr];     // nr of meighbor in reduced grid

          cells->density[p]              = cells->density[p]
                                          + cells_red->density[nr_red];
          cells->temperature_gas[p]      = cells->temperature_gas[p]
                                          + cells_red->temperature_gas[nr_red];
          cells->temperature_gas_prev[p] = cells->temperature_gas_prev[p]
                                          + cells_red->temperature_gas_prev[nr_red];
          cells->thermal_ratio[p]        = cells->thermal_ratio[p]
                                          + cells_red->thermal_ratio[nr_red];
          cells->thermal_ratio_prev[p]   = cells->thermal_ratio_prev[p]
                                          + cells_red->thermal_ratio_prev[nr_red];
        }
      }

      cells->density[p]              = cells->density[p]              / cells->n_neighbors[p];
      cells->temperature_gas[p]      = cells->temperature_gas[p]      / cells->n_neighbors[p];
      cells->temperature_gas_prev[p] = cells->temperature_gas_prev[p] / cells->n_neighbors[p];
      cells->thermal_ratio[p]        = cells->thermal_ratio[p]        / cells->n_neighbors[p];
      cells->thermal_ratio_prev[p]   = cells->thermal_ratio_prev[p]   / cells->n_neighbors[p];

    }

    else
    {
      long nr_red = cells->id[p];   // nr of cell in reduced grid

      cells->density[p]              = cells_red->density[nr_red];
      cells->temperature_gas[p]      = cells_red->temperature_gas[nr_red];
      cells->temperature_gas_prev[p] = cells_red->temperature_gas_prev[nr_red];
      cells->thermal_ratio[p]        = cells_red->thermal_ratio[nr_red];
      cells->thermal_ratio_prev[p]   = cells_red->thermal_ratio_prev[nr_red];
    }

  }
  } // end of OpenMP parallel region


  return (0);

}
