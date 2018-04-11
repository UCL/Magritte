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

int initialize_double_array_with_value (long length, double *array, double value)
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



//
// // initialize_temperature_gas: set gas temperature to a certain initial value
// // --------------------------------------------------------------------------
//
// int initialize_temperature_gas (long ncells, CELLS *cells)
// {
//
// # pragma omp parallel      \
//   shared (ncells, cells)   \
//   default (none)
//   {
//
//   int num_threads = omp_get_num_threads();
//   int thread_num  = omp_get_thread_num();
//
//   long start = (thread_num*NCELLS)/num_threads;
//   long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets
//
//
//   for (long p = start; p < stop; p++)
//   {
//     cells->temperature_gas[p] = 10.0;
//   }
//   } // end of OpenMP parallel region
//
//
//   return (0);
//
// }
//
//
//
//
// // initialize_previous_temperature_gas: set "previous" gas temperature to be 0.9*temperature_gas
// // ---------------------------------------------------------------------------------------------
//
// int initialize_previous_temperature_gas (long ncells, CELLS *cells)
// {
//
// # pragma omp parallel     \
//   shared (ncells, cells)   \
//   default (none)
//   {
//
//   int num_threads = omp_get_num_threads();
//   int thread_num  = omp_get_thread_num();
//
//   long start = (thread_num*NCELLS)/num_threads;
//   long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets
//
//
//   for (long p = start; p < stop; p++)
//   {
//     cells->temperature_gas_prev[p] = 0.9*cells->temperature_gas[p];
//   }
//   } // end of OpenMP parallel region
//
//
//   return (0);
//
// }




// guess_temperature_gas: make a guess for gas temperature based on UV field
// -------------------------------------------------------------------------

int guess_temperature_gas (long ncells, CELLS *cells)
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
    cells->temperature_gas[p] = 10.0*(1.0 + pow(2.0*cells->UV[p], 1.0/3.0));
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_abundances: set abundanceces to initial values
// ---------------------------------------------------------

int initialize_abundances (CELLS *cells, SPECIES species)
{

# pragma omp parallel       \
  shared (cells, species)   \
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
