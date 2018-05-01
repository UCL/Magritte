// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "declarations.hpp"
#include "cells.hpp"
#define NRAYS 2


///  Constructor for CELLS: Allocates memory for cell data
///    @param number_of_cells: number of cells in grid
//////////////////////////////////////////////////////////

CELLS::CELLS (long number_of_cells)
{

  ncells = number_of_cells;


// # if (!FIXED_NCELLS)

    x = new double[ncells];
    y = new double[ncells];
    z = new double[ncells];

    vx = new double[ncells];
    vy = new double[ncells];
    vz = new double[ncells];

    endpoint = new long[ncells*NRAYS];
    Z        = new double[ncells*NRAYS];

    neighbor    = new long[ncells*NRAYS];
    n_neighbors = new long[ncells];

    id      = new long[ncells];
    removed = new bool[ncells];

    boundary = new bool[ncells];
    mirror   = new bool[ncells];

// # endif

}


/// Destructor for CELLS: frees allocated memory
////////////////////////////////////////////////

CELLS::~CELLS ()
{

// # if (!FIXED_NCELLS)

    delete [] x;
    delete [] y;
    delete [] z;

    delete [] vx;
    delete [] vy;
    delete [] vz;

    delete [] endpoint;
    delete [] Z;

    delete [] neighbor;
    delete [] n_neighbors;

    delete [] id;
    delete [] removed;

    delete [] boundary;
    delete [] mirror;

// # endif

}



/// initialize: initialize cells
////////////////////////////////

int CELLS::initialize ()
{
//
// # pragma omp parallel   \
//   default (none)
//   {
//
//   int num_threads = omp_get_num_threads();
//   int thread_num  = omp_get_thread_num();
//
//   long start = (thread_num*ncells)/num_threads;
//   long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets
//
//
//   for (long p = start; p < stop; p++)
//   {
//     x[p] = 0.0;
//     y[p] = 0.0;
//     z[p] = 0.0;
//
//     n_neighbors[p] = 0;
//
//     for (long r = 0; r < NRAYS; r++)
//     {
//       neighbor[RINDEX(p,r)] = 0;
//       endpoint[RINDEX(p,r)] = 0;
//
//       Z[RINDEX(p,r)]           = 0.0;
//
//       for (int y = 0; y < spec_size; y++)
//       {
//         spectrum[NRAYS*ncells*y + RINDEX(p,r)] = 0.0;
//       }
//     }
//
//     vx[p] = 0.0;
//     vy[p] = 0.0;
//     vz[p] = 0.0;
//
//
//
//     id[p] = p;
//
//     removed[p]  = false;
//     boundary[p] = false;
//     mirror[p]   = false;
//   }
//   } // end of OpenMP parallel region
//
//
//   return(0);
//
}
