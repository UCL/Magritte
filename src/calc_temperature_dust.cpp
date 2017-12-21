/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* calc_temperature_dust: Calculate dust temperatures                                     */
/*                                                                                               */
/* (based on dust_t in 3D-PDR)                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Calculate the dust temperature for each particle using the treatment of Hollenbach,           */
/* Takahashi & Tielens (1991, ApJ, 377, 192, eqns 5 & 6) for the heating due to the incident     */
/* FUV photons and the treatment of Meijerink & Spaans (2005, A&A, 436, 397, eqn B.6) for        */
/* heating due to the incident flux of X-ray photons.                                            */
/*                                                                                               */
/* The formula derived by Hollenbach, Takahashi & Tielens (1991) has been modified to include    */
/* the attenuation of the IR radiation. The incident FUV radiation is absorbed and re-emitted    */
/* in the infrared by dust at the surface of the cloud (up to Av ~ 1mag). In the HTT derivation, */
/* this IR radiation then serves as a second heat source for dust deeper into the cloud.         */
/* However, in their treatment, this second re-radiated component is not attenuated with         */
/* distance into the cloud so it is *undiluted* with depth, leading to higher dust temperatures  */
/* deep within the cloud which in turn heat the gas via collisions to unrealistically high       */
/* temperatures. Models with high gas densities and high incident FUV fluxes                     */
/* (e.g. n_H = 10^5 cm-3, X_0 = 10^8 Draine) can produce T_gas ~ 100 K at Av ~ 50 mag!           */
/*                                                                                               */
/* Attenuation of the FIR radiation has therefore been introduced by using an approximation for  */
/* the infrared-only dust temperature from Rowan-Robinson (1980, eqn 30b):                       */
/*                                                                                               */
/* T_dust = T_0*(r/r_0)^(-0.4)                                                                   */
/*                                                                                               */
/* where r_0 is the cloud depth at which T_dust = T_0, corresponding to an A_V of ~ 1 mag, the   */
/* assumed size of the outer region of the cloud that processes the incident FUV radiation and   */
/* then re-emits it in the FIR (see the original HTT 1991 paper for details). This should        */
/* prevent the dust temperature from dropping off too rapidly with distance and maintain a       */
/* larger warm dust region (~50-100 K).                                                          */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_temperature_dust.hpp"



/* calc_temperature_dust: calculate dust temparatures                                     */
/*-----------------------------------------------------------------------------------------------*/

int calc_temperature_dust( double *UV_field, double *rad_surface, double *temperature_dust )
{


  /* Parameters as defined in the paper Hollenbach, Takahashi & Tielens (1991) */

  const double tau_100 = 1.0E-3;                         /* emission optical depth at 100 micron */

  const double nu_0 = 2.65E15;                         /* parameter in the absorption efficiency */


  /* For all grid points */

# pragma omp parallel                                                                             \
  shared( UV_field, rad_surface, temperature_dust )                                               \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long n=start; n<stop; n++){


    /* Contribution to the dust temperature from the local FUV flux and the CMB background */

    temperature_dust[n] = 8.9E-11*nu_0*(1.71*UV_field[n]) + pow(T_CMB, 5);


    for (long r=0; r<NRAYS; r++){


      /* The minimum dust temperature is related to the incident FUV flux along each ray
         Convert the incident FUV flux from Draine to Habing units by multiplying by 1.71 */

      double temperature_min = 12.2*pow(1.71*rad_surface[RINDEX(n,r)], 0.2);


      /* Add the contribution to the dust temperature from the FUV flux incident along this ray */

      if (temperature_min > 0.0){

        temperature_dust[n] = temperature_dust[n]
                              + (0.42-log(3.45E-2*tau_100*temperature_min))
                                *(3.45E-2*tau_100)*pow(temperature_min,6);
      }

    } /* end of r loop over rays */


    temperature_dust[n] = pow(temperature_dust[n], 0.2);


    /* Impose a lower limit on the dust temperature, since values below 10 K can dramatically
       limit the rate of H2 formation on grains (the molecule cannot desorb from the surface) */

    if (temperature_dust[n] < 10.0){

      temperature_dust[n] = 10.0;
    }


    /* Check if the dust temperature is physical */

    if (temperature_dust[n] > 1000.0){

      printf( "(calc_temperature_dust): ERROR," \
              " calculated dust temperature exceeds 1000 K \n" );
    }

  } /* end of n loop over grid points */
  } /* end of OpenMP parallel region */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
