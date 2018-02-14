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


// reduce: reduce number of cells, return resulting number of cells
// ----------------------------------------------------------------

long reduce (long ncells, CELL *cell, double min_density_change,
             double x_min, double x_max, double y_min, double y_max, double z_min, double z_max)
{


  // Crop grid

  std::cout << "  Cropping input grid...\n";

  crop (NCELLS, cell, x_min, x_max, y_min, y_max, z_min, z_max);


  // Reduce grid

  std::cout << "  Reducing input grid...\n";

  density_reduce (NCELLS, cell, min_density_change);


  // Set id's to relate grid and reduced grid, get ncells_red

  std::cout << "  Setting id's for reduced grid...\n";

  long ncells_red = set_ids (NCELLS, cell);


  return ncells_red;

}




// crop: crop spatial range of data
// --------------------------------

int crop (long ncells, CELL *cell,
          double x_min, double x_max, double y_min, double y_max, double z_min, double z_max)
{

# pragma omp parallel                                               \
  shared (ncells, cell, x_min, x_max, y_min, y_max, z_min, z_max)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    if (    (x_min > cell[p].x) || (cell[p].x > x_max)
         || (y_min > cell[p].y) || (cell[p].y > y_max)
         || (z_min > cell[p].z) || (cell[p].z > z_max) )
    {
      cell[p].removed = true;
    }

  }
  } // end of OpenMP parallel region


  return (0);

}




// density_reduce: reduce number of cell in regions of constant density
// --------------------------------------------------------------------

int density_reduce (long ncells, CELL *cell, double min_density_change)
{

  // Note that this loop cannot be paralellized !

  for (long p = 0; p < NCELLS; p++)
  {
    if (!cell[p].removed)
    {

      double density_c = cell[p].density;   // density of current cell

      cell[p].removed  = true;              // assume cell can be removed


      // Check whether cell can indeed be removed

      for (int n = 0; n < cell[p].n_neighbors; n++)
      {
        long nr = cell[p].neighbor[n];

        double rel_density_change = 2.0*fabs(cell[nr].density - density_c)
                                        / (cell[nr].density + density_c);


        // Do not remove if density changes too much or neighbor was removed
        // The latter to avoid large gaps being formed

        if ( (rel_density_change > min_density_change) || cell[nr].removed)
        {
          cell[p].removed = false;
        }
      }

    }
  }


  return (0);

}




// set_ids: determine cell numbers in the reduced grid, return nr of reduced cells
// -------------------------------------------------------------------------------

long set_ids (long ncells, CELL *cell)
{

  long cell_id_reduced = 0;


  // Store nr of cell in reduced grid in id's of full grid
  // Note that this loop cannot be paralellized !

  for (long p = 0; p < NCELLS; p++)
  {
    if (cell[p].removed)
    {
      cell[p].id = -1;
    }

    else
    {
      cell[p].id = cell_id_reduced;

      cell_id_reduced++;
    }
  }


  // cell_id_reduced now equals total number of cells in reduced grid

  return cell_id_reduced;

}




// initialized_reduced_grid: initialize reduced grid
// -------------------------------------------------

int initialize_reduced_grid (long ncells_red, CELL *cell_red, long ncells, CELL *cell)
{

# pragma omp parallel                           \
  shared (ncells_red, cell_red, ncells, cell)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    if (!cell[n].removed)
    {
      long nr = cell[n].id;   // nr of cell n in reduced grid

      cell_red[nr].x = cell[n].x;
      cell_red[nr].y = cell[n].y;
      cell_red[nr].z = cell[n].z;

      cell_red[nr].vx = cell[n].vx;
      cell_red[nr].vy = cell[n].vy;
      cell_red[nr].vz = cell[n].vz;

      cell_red[nr].density = cell[n].density;

      cell_red[nr].temperature.gas      = cell[n].temperature.gas;
      cell_red[nr].temperature.dust     = cell[n].temperature.dust;
      cell_red[nr].temperature.gas_prev = cell[n].temperature.gas_prev;
    }
  }
  } // end of OpenMP parallel region


  return (0);
}




// interpolate: interpolate reduced grid back to original grid
// -----------------------------------------------------------

int interpolate (long ncells_red, CELL *cell_red, long ncells, CELL *cell)
{

# pragma omp parallel                           \
  shared (ncells_red, cell_red, ncells, cell)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    if (cell[p].removed)
    {

      // Take average of neighbors

      cell[p].density              = 0.0;
      cell[p].temperature.gas      = 0.0;
      cell[p].temperature.gas_prev = 0.0;


      for (int n = 0; n < cell[p].n_neighbors; n++)
      {
        long nr = cell[p].neighbor[n];   // nr of neighbor in grid

        if (!cell[nr].removed)
        {
          long nr_red = cell[nr].id;     // nr of meighbor in reduced grid

          cell[p].density              = cell[p].density
                                         + cell_red[nr_red].density;
          cell[p].temperature.gas      = cell[p].temperature.gas
                                         + cell_red[nr_red].temperature.gas;
          cell[p].temperature.gas_prev = cell[p].temperature.gas_prev
                                         + cell_red[nr_red].temperature.gas_prev;
        }
      }

      cell[p].density              = cell[p].density / cell[p].n_neighbors;
      cell[p].temperature.gas      = cell[p].temperature.gas / cell[p].n_neighbors;
      cell[p].temperature.gas_prev = cell[p].temperature.gas_prev / cell[p].n_neighbors;

    }

    else
    {
      long nr_red = cell[p].id;   // nr of cell in reduced grid

      cell[p].density              = cell_red[nr_red].density;
      cell[p].temperature.gas      = cell_red[nr_red].temperature.gas;
      cell[p].temperature.gas_prev = cell_red[nr_red].temperature.gas_prev;
    }

  }
  } // end of OpenMP parallel region


  return (0);

}
