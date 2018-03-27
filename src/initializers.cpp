// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "initializers.hpp"



// initialize_int_array: set all entries of array of ints equal to zero
// --------------------------------------------------------------------

int initialize_int_array (long length, int *array)
{

# pragma omp parallel      \
  shared (array, length)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    array[i] = 0;
  }
  } // end of OpenMP parallel region


  return(0);

}


// initialize_long_array: set all entries of array of longs equal to zero
// ----------------------------------------------------------------------

int initialize_long_array (long length, long *array)
{

# pragma omp parallel      \
  shared (array, length)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    array[i] = 0;
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_double_array: sets all entries of array of doubles equal to zero
// ---------------------------------------------------------------------------

int initialize_double_array (long length, double *array)
{

# pragma omp parallel      \
  shared (array, length)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    array[i] = 0.0;
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_double_array_with: sets entries of first array of doubles equal to second
// ------------------------------------------------------------------------------------

int initialize_double_array_with (long length, double *array1, double *array2)
{

# pragma omp parallel               \
  shared (array1, array2, length)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    array1[i] = array2[i];
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_double_array_with_scale: sets first array of doubles equal to second with scale
// ------------------------------------------------------------------------------------------

int initialize_double_array_with_scale (long length, double *array1, double *array2, double scale)
{

# pragma omp parallel                      \
  shared (array1, array2, length, scale)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    array1[i] = scale*array2[i];
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_double_array_with_value: sets entries of array of doubles equal to value
// -----------------------------------------------------------------------------------

int initialize_double_array_with_value (long length, double value, double *array)
{

# pragma omp parallel             \
  shared (array, value, length)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    array[i] = value;
  }
  } // end of OpenMP parallel region


  return(0);

}




// initialize_char_array: sets all entries of array of chars equal to 'i'
// ------------------------------------------ ---------------------------

int initialize_char_array (long length, char *array)
{

# pragma omp parallel      \
  shared (array, length)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    array[i] = 'i';
  }
  } // end of OpenMP parallel region


  return(0);

}




// initialize_cells: initialize the cell array
// -------------------------------------------

int initialize_cells (long ncells, CELLS *cells)
{

# pragma omp parallel      \
  shared (ncells, cells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cells->x[p] = 0.0;
    cells->y[p] = 0.0;
    cells->z[p] = 0.0;

    cells->n_neighbors[p] = 0;

    for (long r = 0; r < NRAYS; r++)
    {
      cells->neighbor[RINDEX(p,r)] = 0;
      cells->endpoint[RINDEX(p,r)] = 0;

      cells->Z[RINDEX(p,r)]           = 0.0;
      cells->intensity[RINDEX(p,r)]   = 0.0;
      cells->column[RINDEX(p,r)]      = 0.0;
      cells->rad_surface[RINDEX(p,r)] = 0.0;
      cells->AV[RINDEX(p,r)]          = 0.0;
    }

    cells->vx[p] = 0.0;
    cells->vy[p] = 0.0;
    cells->vz[p] = 0.0;

    cells->density[p] = 0.0;

    cells->UV[p] = 0.0;

    for (int s = 0; s < NSPEC; s++)
    {
      cells->abundance[SINDEX(p,s)] = 0.0;
    }

    for (int e = 0; e < NREAC; e++)
    {
      cells->rate[READEX(p,e)] = 0.0;
    }

    for (int l = 0; l < TOT_NLEV; l++)
    {
      cells->pop[LINDEX(p,l)] = 0.0;
    }

    for (int k = 0; k < TOT_NRAD; k++)
    {
      cells->mean_intensity[KINDEX(p,k)] = 0.0;
    }

    cells->temperature_gas[p]      = 0.0;
    cells->temperature_dust[p]     = 0.0;
    cells->temperature_gas_prev[p] = 0.0;

    cells->thermal_ratio[p]      = 1.0;
    cells->thermal_ratio_prev[p] = 1.1;

    cells->id[p] = p;

    cells->removed[p]  = false;
    cells->boundary[p] = false;
    cells->mirror[p]   = false;
  }
  } // end of OpenMP parallel region


  initialize_temperature_gas (NCELLS, cells);
  initialize_previous_temperature_gas (NCELLS, cells);


  return(0);

}




// initialize_cell_id: initialize the cell id's
// --------------------------------------------

int initialize_cell_id (long ncells, CELLS *cells)
{

# pragma omp parallel      \
  shared (ncells, cells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cells->id[p] = p;
  }
  } // end of OpenMP parallel region


  return(0);

}




// initialize_temperature_gas: set gas temperature to a certain initial value
// --------------------------------------------------------------------------

int initialize_temperature_gas (long ncells, CELLS *cells)
{

# pragma omp parallel      \
  shared (ncells, cells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cells->temperature_gas[p] = 10.0;
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_previous_temperature_gas: set "previous" gas temperature to be 0.9*temperature_gas
// ---------------------------------------------------------------------------------------------

int initialize_previous_temperature_gas (long ncells, CELLS *cells)
{

# pragma omp parallel     \
  shared (ncells, cells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cells->temperature_gas_prev[p] = 0.9*cells->temperature_gas[p];
  }
  } // end of OpenMP parallel region


  return (0);

}




// guess_temperature_gas: make a guess for gas temperature based on UV field
// -------------------------------------------------------------------------

int guess_temperature_gas (long ncells, CELLS *cells)
{

# pragma omp parallel     \
  shared (ncells, cells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    cells->temperature_gas[p] = 100.0*(1.0 + pow(2.0*cells->UV[p], 1.0/3.0));
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_abundances: set abundanceces to initial values
// ---------------------------------------------------------

int initialize_abundances (long ncells, CELLS *cells, SPECIES species)
{

# pragma omp parallel               \
  shared (ncells, cells, species)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    for (int s = 0; s < NSPEC; s++)
    {
      cells->abundance[SINDEX(p,s)] = species.initial_abundance[s];
    }
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_bool: initialize a boolean variable
// ----------------------------------------------

int initialize_bool (long length, bool value, bool *variable)
{

# pragma omp parallel                \
  shared (length, variable, value)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;   // Note brackets


  for (long i = start; i < stop; i++)
  {
    variable[i] = value;
  }
  } // end of OpenMP parallel region


  return (0);

}
