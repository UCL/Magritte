/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* calc_C_coeff: Calculate C coefficients for all temperatures by interpolation from datafiles   */
/*                                                                                               */
/* (based on find_Ccoeff in 3D-PDR)                                                              */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <iostream>
#include <stdio.h>
#include <math.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_C_coeff.hpp"
#include "initializers.hpp"



/* calc_C_coeff: calculates the collisional coefficients (C_ij) from the line data               */
/*-----------------------------------------------------------------------------------------------*/

int calc_C_coeff( GRIDPOINT *gridpoint, double *C_data, double *coltemp, int *icol, int *jcol,
                  double *temperature_gas, double *weight, double *energy, double *C_coeff,
                  long gridp, int lspec )
{


  double step;                                                    /* (linear) interpolation step */


  // printf("(calc_C_coeff): intput C_data = \n");

  // for (par=0; par<ncolpar[lspec]; par++){

  //   for (int ctran=0; ctran<ncoltran[LSPECPAR(lspec,par)]; ctran++){

  //     for (int ctemp=0; ctemp<ncoltemp[LSPECPAR(lspec,par)]; ctemp++){

  //       printf( "  %.2lE", C_data[LSPECPARTRANTEMP(lspec,par,ctran,ctemp)] );
  //     }

  //     printf("\n");
  //   }

  //   printf("\n");
  // }




  /* Calculate H2 ortho/para fraction at equilibrium for given temperature */

  double frac_H2_para  = 0.0;                                             /* fraction of para-H2 */
  double frac_H2_ortho = 0.0;                                            /* fraction of ortho-H2 */


  if (species[H2_nr].abn[gridp] > 0.0){

    frac_H2_para  = 1.0 / (1.0 + 9.0*exp(-170.5/temperature_gas[gridp]));
    frac_H2_ortho = 1.0 - frac_H2_para;
  }



  /* Initialize C_coeff */

  initialize_double_array(C_coeff, TOT_NLEV2);



  /* For all collision partners */

  for (int par=0; par<ncolpar[lspec]; par++){


    /* Get the number of the species corresponding to the collision partner */

    int spec = spec_par[lspec,par];


    /* Find the available temperatures closest to the actual tamperature */

    int tindex_low  = -1;                       /* index of temperature below actual temperature */
    int tindex_high = -1;                       /* index of temperature above actual temperature */


    /* Find the data corresponding to the temperatures above and below the actual temperature */

    for (int tindex=0; tindex<ncoltemp[LSPECPAR(lspec,par)]; tindex++ ){

      if (temperature_gas[gridp] < coltemp[LSPECPARTEMP(lspec,par,tindex)]){

        tindex_low  = tindex-1;
        tindex_high = tindex;

        break;
      }
    }

    if (tindex_high == -1){

      tindex_high = tindex_low = ncoltemp[LSPECPAR(lspec,par)]-1;
    }

    if (tindex_high == 0){

      tindex_high = tindex_low = 0;
    }


    double C_T_low[nlev[lspec]*nlev[lspec]];

    initialize_double_array(C_T_low, nlev[lspec]*nlev[lspec]);

    double C_T_high[nlev[lspec]*nlev[lspec]];

    initialize_double_array(C_T_high, nlev[lspec]*nlev[lspec]);


    for (int ckr=0; ckr<ncoltran[LSPECPAR(lspec,par)]; ckr++){

      int i = icol[LSPECPARTRAN(lspec,par,ckr)];
      int j = jcol[LSPECPARTRAN(lspec,par,ckr)];

      C_T_low[LINDEX(i,j)]  = C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_low)];
      C_T_high[LINDEX(i,j)] = C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_high)];
    }


    /* Calculate the reverse (excitation) rate coefficients from detailed balance, if not given
       i.e. C_ji = C_ij * g_i/g_j * exp( -(E_i-E_j)/ (kb T) ) */

    for (int ckr=0; ckr<ncoltran[LSPECPAR(lspec,par)]; ckr++){

      int i = icol[LSPECPARTRAN(lspec,par,ckr)];
      int j = jcol[LSPECPARTRAN(lspec,par,ckr)];

      int l_i = LSPECLEV(lspec,i);
      int l_j = LSPECLEV(lspec,j);


      if ( C_T_low[LINDEX(j,i)] == 0.0 && C_T_low[LINDEX(i,j)] != 0.0 ){

        C_T_low[LINDEX(j,i)] = C_T_low[LINDEX(i,j)] * weight[l_i]/weight[l_j]
                               * exp( -(energy[l_i] - energy[l_j])
                                       /(KB*coltemp[LSPECPARTEMP(lspec,par,tindex_low)]) );
      }


      if ( C_T_high[LINDEX(j,i)] == 0.0 && C_T_high[LINDEX(i,j)] != 0.0 ){

        C_T_high[LINDEX(j,i)] = C_T_high[LINDEX(i,j)] * weight[l_i]/weight[l_j]
                                * exp( -(energy[l_i] - energy[l_j])
                                        /(KB*coltemp[LSPECPARTEMP(lspec,par,tindex_high)]) );
      }
    }


    /* Calculate the (linear) interpolation step */

    double step = 0.0;                                            /* (linear) interpolation step */

    if (tindex_high != tindex_low){

      step = (temperature_gas[gridp] - coltemp[LSPECPARTEMP(lspec,par,tindex_low)])
              / ( coltemp[LSPECPARTEMP(lspec,par,tindex_high)]
                  - coltemp[LSPECPARTEMP(lspec,par,tindex_low)] );
    }

    // printf("(calc_C_coeff): step %.2lf \n", step);


    /* For all C matrix elements */

    for (int i=0; i<nlev[lspec]; i++){

      for (int j=0; j<nlev[lspec]; j++){


        /* Make a linear interpolation for C in the temperature */

        double C_tmp = C_T_low[LINDEX(i,j)] + (C_T_high[LINDEX(i,j)]-C_T_low[LINDEX(i,j)]) * step;

        // printf( "(calc_C_coeff): C_tmp = %.2lE \t tindex_low %d \t tindex_high %d \n",
        //         C_tmp, tindex_low, tindex_high );



        /* Weigh contributions to C by abundance */

        double abundance = gridpoint[gridp].density * species[spec].abn[gridp];

        if ( ortho_para[LSPECPAR(lspec,par)] == 'o' ){

          abundance = abundance * frac_H2_ortho;
        }

        else if ( ortho_para[LSPECPAR(lspec,par)] == 'p' ){

          abundance = abundance * frac_H2_para;
        }

        C_coeff[LSPECLEVLEV(lspec,i,j)] = C_coeff[LSPECLEVLEV(lspec,i,j)] + C_tmp*abundance;

      }

    }

  } /* end of par loop ovrer collision partners */


  // printf("(calc_C_coeff): output [C_ij]= \n");

  // for (i=0; i<nlev[lspec]; i++){

  //   for (j=0; j<nlev[lspec]; j++){

  //     printf( "\t %.2lE", C_coeff[LSPECLEVLEV(lspec,i,j)] );
  //   }

  //   printf("\n");
  // }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
