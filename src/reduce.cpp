// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "reduce.hpp"
#include "initializers.hpp"
#include "write_txt_tools.hpp"


// reduce: reduce number of cells
// ------------------------------

int reduce (long ncells, CELL *cell, double threshold,
            double x_min, double x_max, double y_min, double y_max, double z_min, double z_max)
{

  // Make sure cell id's are initialized

  initialize_cell_id (cell, NCELLS);


  // Crop grid

  std::cout << "  Cropping input grid...\n";

  crop (NCELLS, cell, x_min, x_max, y_min, y_max, z_min, z_max);


  // write cropped grid as .txt file

  std::cout << "  Writing .txt grid...\n";

  write_grid ("cropped", NCELLS, cell);


  // Reduce grid

  std::cout << "  Reducing input grid...\n";

  density_reduce (NCELLS, cell, threshold);


  return (0);

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
      cell[p].id = -1;
    }

  }
  } // end of OpenMP parallel region


  return (0);

}




// density_reduce: reduce number of cell in regions of constant density
//---------------------------------------------------------------------

int density_reduce (long ncells, CELL *cell, double threshold)
{

# pragma omp parallel                \
  shared (ncells, cell, threshold)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    if (cell[p].id == p)   // if cell[p].id == p this means cell p is not removed
    {

      double density_local = cell[p].density;   // density of current cell

      bool remove_neighbors = true;   // assume neighbors can be removed


      // Check whether neighbors can be removed

      for (int n = 0; n < cell[p].n_neighbors; n++)
      {
        long nr = cell[p].neighbor[n];

        double rel_density_change = fabs(cell[nr].density - cell[p].density) /cell[p].density;

        if (rel_density_change > threshold)
        {
          remove_neighbors = false;
        }
      }


      // Remove neighbors if possible

      if (remove_neighbors)
      {
        for (int n = 0; n < cell[p].n_neighbors; n++)
        {
          long nr = cell[p].neighbor[n];

          cell[nr].id = p;
        }
      }

    } // if cell[n].id == n

  }
  } // end of OpenMP parallel region


  return (0);

}
