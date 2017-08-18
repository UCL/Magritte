/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Initializers.c: Initialization functions for all (linearized) arrays                          */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <math.h>

#include "declarations.hpp"
#include "initializers.hpp"



/* initialize_int_array: sets all entries of the linearized array of ints equal to zero          */
/*-----------------------------------------------------------------------------------------------*/

int initialize_int_array(int *array, long length)
{


  for(int i=0; i<length; i++){

    array[i] = 0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/



/* initialize_long_array: sets all entries of the linearized array of longs equal to zero        */
/*-----------------------------------------------------------------------------------------------*/

int initialize_long_array(long *array, long length)
{


  for(int i=0; i<length; i++){

    array[i] = 0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_double_array: sets all entries of the linearized array of doubles equal to zero    */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array(double *array, long length)
{


  for(int i=0; i<length; i++){

    array[i] = 0.0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_double_array_with: sets entries of the first array of doubles equal to the second  */
/*-----------------------------------------------------------------------------------------------*/

int initialize_double_array_with(double *array1, double *array2, long length)
{


  for(int i=0; i<length; i++){

    array1[i] = array2[i];
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_char_array: sets all entries of the linearized array of doubles equal to 'i'       */
/*-----------------------------------------------------------------------------------------------*/

int initialize_char_array(char *array, long length)
{


  for(int i=0; i<length; i++){

    array[i] = 0.0;
  }


  return(0);

}


/*-----------------------------------------------------------------------------------------------*/





/* initialize_evalpoint: sets all entries of the linearized array equal to zero or false         */
/*-----------------------------------------------------------------------------------------------*/

int initialize_evalpoint(EVALPOINT *evalpoint)
{


    for (int n1=0; n1<NGRID; n1++){

      for (int n2=0; n2<NGRID; n2++){

        evalpoint[GINDEX(n1,n2)].dZ  = 0.0;
        evalpoint[GINDEX(n1,n2)].Z   = 0.0;
        evalpoint[GINDEX(n1,n2)].vol = 0.0;

        evalpoint[GINDEX(n1,n2)].ray = 0;
        evalpoint[GINDEX(n1,n2)].nr  = 0;

        evalpoint[GINDEX(n1,n2)].eqp = 0;

        evalpoint[GINDEX(n1,n2)].onray = false;
      }
    }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_temperature_gas: set the gas temperature to a certain initial value                */
/*-----------------------------------------------------------------------------------------------*/

int initialize_temperature_gas(double *temperature_gas)
{


  for (int n=0; n<NGRID; n++){

    temperature_gas[n] = 10.0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* initialize_level_populations: sets pops of all line species to the thermal equilibrium value  */
/*-----------------------------------------------------------------------------------------------*/

int initialize_level_populations(double *energy, double *temperature_gas, double *pop)
{


  for (int lspec=0; lspec<NLSPEC; lspec++){

    for (int n=0; n<NGRID; n++){

      for (int i=0; i<nlev[lspec]; i++){


        long p_i = LSPECGRIDLEV(lspec,n,i);


        pop[p_i] = exp( -HH*CC*energy[LSPECLEV(lspec,i)] / (KB*temperature_gas[n]) );


        /* Avoid too small numbers */

        if (pop[p_i] < POP_LOWER_LIMIT){

          pop[p_i] = 0.0;
        }

      }
    }
  }


  return(0);
}

/*-----------------------------------------------------------------------------------------------*/
