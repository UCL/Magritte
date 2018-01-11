// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "initializers.hpp"



// initialize_int_array: set all entries of array of ints equal to zero
// --------------------------------------------------------------------

int initialize_int_array (int *array, long length)
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

int initialize_long_array (long *array, long length)
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

int initialize_double_array (double *array, long length)
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

int initialize_double_array_with (double *array1, double *array2, long length)
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




// initialize_double_array_with_value: sets entries of array of doubles equal to value
// -----------------------------------------------------------------------------------

int initialize_double_array_with_value (double *array, double value, long length)
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

int initialize_char_array (char *array, long length)
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

int initialize_cells (CELL *cell, long ncells)
{


# pragma omp parallel     \
  shared (cell, ncells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    cell[n].x = 0.0;
    cell[n].y = 0.0;
    cell[n].z = 0.0;

    cell[n].n_neighbors = 0;

    for (long ray = 0; ray < NRAYS; ray++)
    {
      cell[n].neighbor[ray] = 0;
    }

    cell[n].vx = 0.0;
    cell[n].vy = 0.0;
    cell[n].vz = 0.0;

    cell[n].density = 0.0;

    for (int spec = 0; spec < NSPEC; spec++)
    {
      cell[n].abundance[spec] = 0.0;
    }

    for (int reac = 0; reac < NREAC; reac++)
    {
      cell[n].rate[reac] = 0.0;
    }

    cell[n].temperature.gas      = 0.0;
    cell[n].temperature.dust     = 0.0;
    cell[n].temperature.gas_prev = 0.0;

    cell[n].id = n;

    cell[n].removed = false;
  }
  } // end of OpenMP parallel region


  return(0);

}




// initialize_cell_id: initialize the cell id's
// --------------------------------------------

int initialize_cell_id (CELL *cell, long ncells)
{


# pragma omp parallel     \
  shared (cell, ncells)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    cell[n].id = n;
  }
  } // end of OpenMP parallel region


  return(0);

}




// initialize_temperature_gas: set gas temperature to a certain initial value
// --------------------------------------------------------------------------

int initialize_temperature_gas (long ncells, CELL *cell)
{

# pragma omp parallel   \
  shared (cell)         \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    cell[n].temperature.gas = 10.0;
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_previous_temperature_gas: set "previous" gas temperature to be 0.9*temperature_gas
// ---------------------------------------------------------------------------------------------

int initialize_previous_temperature_gas (long ncells, CELL *cell)
{

# pragma omp parallel   \
  shared (cell)         \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    cell[n].temperature.gas_prev = 0.9*cell[n].temperature.gas;
  }
  } // end of OpenMP parallel region


  return (0);

}




// guess_temperature_gas: make a guess for gas temperature based on UV field
// -------------------------------------------------------------------------

int guess_temperature_gas (long ncells, CELL *cell, double *UV_field)
{

# pragma omp parallel       \
  shared (cell, UV_field)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    cell[n].temperature.gas = 10.0*(1.0 + pow(2.0*UV_field[n], 1.0/3.0));
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_abundances: set abundanceces to initial values
// ---------------------------------------------------------

int initialize_abundances (long ncells, CELL *cell, SPECIES *species)
{

# pragma omp parallel              \
  shared (ncells, cell, species)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    for (int spec = 0; spec < NSPEC; spec++)
    {
      cell[n].abundance[spec] = species[spec].initial_abundance;
    }
  }
  } // end of OpenMP parallel region


  return (0);

}




// initialize_bool: initialize a boolean variable
// ----------------------------------------------

int initialize_bool (bool value, bool *variable, long length)
{

# pragma omp parallel                \
  shared (value, length, variable)   \
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
