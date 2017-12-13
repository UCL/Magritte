/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* initializers.cpp: Initialization functions for all (linearized) arrays                        */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "initializers.hpp"



/* initialize_int_array: sets all entries of the linearized array of ints equal to zero          */
/*-----------------------------------------------------------------------------------------------*/

int initialize_int_array(int *array, long length)
{


# pragma omp parallel                                                                             \
  shared( array, length )                                                                         \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;      /* Note the brackets are important here */


  for (long i=start; i<stop; i++){

    array[i] = 0;
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/



/* initialize_long_array: sets all entries of the linearized array of longs equal to zero        */
/*-----------------------------------------------------------------------------------------------*/

int initialize_long_array(long *array, long length)
{


# pragma omp parallel                                                                             \
  shared( array, length )                                                                         \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;      /* Note the brackets are important here */


  for (long i=start; i<stop; i++){

    array[i] = 0;
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_double_array: sets all entries of the linearized array of doubles equal to zero    */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array(double *array, long length)
{


# pragma omp parallel                                                                             \
  shared( array, length )                                                                         \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;      /* Note the brackets are important here */


  for (long i=start; i<stop; i++){

    array[i] = 0.0;
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_double_array_with: sets entries of the first array of doubles equal to the second  */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array_with(double *array1, double *array2, long length)
{


# pragma omp parallel                                                                             \
  shared( array1, array2, length )                                                                \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;      /* Note the brackets are important here */


  for (long i=start; i<stop; i++){

    array1[i] = array2[i];
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_double_array_with_value: sets entries of the array of doubles equal to value       */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array_with_value(double *array, double value, long length)
{


# pragma omp parallel                                                                             \
  shared( array, value, length )                                                                  \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;      /* Note the brackets are important here */


  for (long i=start; i<stop; i++){

    array[i] = value;
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_char_array: sets all entries of the linearized array of doubles equal to 'i'       */
/*-----------------------------------------------------------------------------------------------*/

int initialize_char_array(char *array, long length)
{


# pragma omp parallel                                                                             \
  shared( array, length )                                                                         \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;      /* Note the brackets are important here */


  for (long i=start; i<stop; i++){

    array[i] = 'i';
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}


/*-----------------------------------------------------------------------------------------------*/





// /* initialize_evalpoint: sets all entries of the linearized array equal to zero or false         */
// /*-----------------------------------------------------------------------------------------------*/
//
// int initialize_evalpoint(EVALPOINT *evalpoint, long length)
// {
//
//
//     for (long n=0; n<length; n++){
//
//       evalpoint[n].dZ  = 0.0;
//       evalpoint[n].Z   = 0.0;
//       evalpoint[n].vol = 0.0;
//
//       evalpoint[n].ray = 0;
//       evalpoint[n].nr  = 0;
//
//       evalpoint[n].eqp = 0;
//
//       evalpoint[n].onray = false;
//     }
//
//
//   return EXIT_SUCCESS;
//
// }
//
// /*-----------------------------------------------------------------------------------------------*/





/* initialize_temperature_gas: set the gas temperature to a certain initial value                */
/*-----------------------------------------------------------------------------------------------*/

int initialize_temperature_gas(double *temperature_gas)
{


# pragma omp parallel                                                                             \
  shared( temperature_gas )                                                                       \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long n=start; n<stop; n++){

    temperature_gas[n] = 10.0;
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_previous_temperature_gas: set the "previous" gas temperature 0.9*temperature_gas   */
/*-----------------------------------------------------------------------------------------------*/

int initialize_previous_temperature_gas(double *previous_temperature_gas, double *temperature_gas)
{


# pragma omp parallel                                                                             \
  shared( previous_temperature_gas, temperature_gas )                                             \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long n=start; n<stop; n++){

    previous_temperature_gas[n] = 0.9*temperature_gas[n];
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* guess_temperature_gas: make a guess for the gas temperature based on the UV field             */
/*-----------------------------------------------------------------------------------------------*/

int guess_temperature_gas(double *UV_field, double *temperature_gas)
{


# pragma omp parallel                                                                             \
  shared( UV_field, temperature_gas )                                                             \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long n=start; n<stop; n++){

    temperature_gas[n] = 10.0*( 1.0 + pow(2.0*UV_field[n], 1.0/3.0) );
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_abn: set the abundanceces to the initial values                                    */
/*-----------------------------------------------------------------------------------------------*/

int initialize_abn(double *initial_abn)
{


# pragma omp parallel                                                                             \
  shared( initial_abn, species )                                                                  \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long n=start; n<stop; n++){

    for (int spec=0; spec<NSPEC; spec++){

      species[spec].abn[n] = initial_abn[spec];
    }
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_bool: initialize a boolean variable                                                */
/*-----------------------------------------------------------------------------------------------*/

int initialize_bool(bool value, long length, bool *variable)
{


# pragma omp parallel                                                                             \
  shared( value, length, variable )                                                               \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*length)/num_threads;
  long stop  = ((thread_num+1)*length)/num_threads;      /* Note the brackets are important here */


  for (long i=start; i<stop; i++){

    variable[i] = value;
  }
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_level_populations: sets pops of all line species to the thermal equilibrium value  */
/*-----------------------------------------------------------------------------------------------*/

// int initialize_level_populations(double *energy, double *temperature_gas, double *pop)
// {
//
//
//   for (int lspec=0; lspec<NLSPEC; lspec++){
//
//     for (int n=0; n<NGRID; n++){
//
//       for (int i=0; i<nlev[lspec]; i++){
//
//
//         long p_i = LSPECGRIDLEV(lspec,n,i);
//
//
//         pop[p_i] = exp( -HH*CC*energy[LSPECLEV(lspec,i)] / (KB*temperature_gas[n]) );
//
//
//         /* Avoid too small numbers */
//
//         if (pop[p_i] < POP_LOWER_LIMIT){
//
//           pop[p_i] = 0.0;
//         }
//
//       }
//     }
//   }
//
//
//   return EXIT_SUCCESS;
// }

/*-----------------------------------------------------------------------------------------------*/
