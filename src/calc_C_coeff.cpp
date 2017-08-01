/* Frederik De Ceuster - University College London                                               */
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



#include <stdio.h>
#include <math.h>

#include "declarations.hpp"
#include "calc_C_coeff.hpp"



/* calc_C_coeff: calculates the collisional coefficients (C_ij) from the line data               */
/*-----------------------------------------------------------------------------------------------*/

void calc_C_coeff( double *C_data, double *coltemp, int *icol, int *jcol, double *temperature,
                   double *weight, double *energy, double *C_coeff, long gridp, int lspec )
{

  int par;                                                      /* index for a collision partner */

  int H2_nr;                                                   /* species nr corresponding to H2 */

  int spec;                                                                     /* species index */

  int tindex;                                                               /* temperature index */
  int tindex_low;                               /* index of temperature below actual temperature */
  int tindex_high;                              /* index of temperature above actual temperature */

  int ckr;                                                       /* collisional transition index */

  int i, j;                                                                     /* level indices */

  double frac_H2_para;                                                    /* fraction of para-H2 */
  double frac_H2_ortho;                                                  /* fraction of ortho-H2 */

  double abundance;                                        /* abundance of the collision partner */

  double step;                                                    /* (linear) interpolation step */

  double C_tmp;                                                         /* temporary value for C */

  int max_ncoltran;                   /* maximum number of collisional transitions for a partner */
  int par_max_ncoltran;                /* partner with maximum number of collisional transitions */



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



  /* Use species_tools to find the number corresponding to H2 */

  int get_species_nr(string name);

  H2_nr = get_species_nr("H2");



  /* Calculate H2 ortho/para fraction at equilibrium for given temperature */

  frac_H2_para = 0.0;
  frac_H2_ortho = 0.0;


  if (species[H2_nr].abn[gridp] > 0){

    frac_H2_para = 1.0 / (1.0 + 9.0*exp(-170.5/temperature[gridp]));
    frac_H2_ortho = 1.0 - frac_H2_para;
  }


  max_ncoltran = 0;
  par_max_ncoltran = 0;


  /* For all collision partners */

  for (par=0; par<ncolpar[lspec]; par++){


    /* Get the number of the species corresponding to the collision partner */

    spec = spec_par[lspec,par];

    /* Find max number of collisional transitions for a partner */

    if (ncoltran[LSPECPAR(lspec,par)] > max_ncoltran){

      max_ncoltran = ncoltran[par];
      par_max_ncoltran = par;
    }


    /* Find the available temperatures closest to the actual tamperature */

    tindex_low = -1;
    tindex_high = -1;


    /* for all available temperatures */

    for (tindex=0; tindex<ncoltemp[LSPECPAR(lspec,par)]; tindex++ ){

      if (temperature[gridp] < coltemp[LSPECPARTEMP(lspec,par,tindex)]){

        tindex_low = tindex-1;
        tindex_high  = tindex;

        break;
      }
    }


    if (tindex_high == -1){

      tindex_high = tindex_low = ncoltemp[LSPECPAR(lspec,par)]-1;
    }


    if (tindex_high == 0){

      tindex_high = tindex_low = 0;
    }



    /* Calculate the (linear) interpolation step */

    if (tindex_high == tindex_low){

      step = 0.0;
    }
    else {

      step = (temperature[gridp] - coltemp[LSPECPARTEMP(lspec,par,tindex_low)])
              / ( coltemp[LSPECPARTEMP(lspec,par,tindex_high)]
                  - coltemp[LSPECPARTEMP(lspec,par,tindex_low)] );
    }

    // printf("(calc_C_coeff): step %.2lf \n", step);


    /* For all collisional transitions */

    for (ckr=0; ckr<ncoltran[LSPECPAR(lspec,par)]; ckr++){

      i = icol[LSPECPARTRAN(lspec,par,ckr)];
      j = jcol[LSPECPARTRAN(lspec,par,ckr)];


      /* Make a linear interpolation for C in the temperature */

      C_tmp = C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_low)]
              + ( C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_high)]
                  - C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_low)] )*step;

      // printf( "(calc_C_coeff): C_tmp = %.2lE \t tindex_low %d \t tindex_high %d \n",
      //         C_tmp, tindex_low, tindex_high );



      /* Weigh contributions to C by abundance */

      abundance = species[spec].abn[gridp];

      if ( ortho_para[LSPECPAR(lspec,par)] == 'o' ){

        abundance = abundance * frac_H2_ortho;
      }

      else if ( ortho_para[LSPECPAR(lspec,par)] == 'p' ){

        abundance = abundance * frac_H2_para;
      }

      C_coeff[LSPECLEVLEV(lspec,i,j)] = C_coeff[LSPECLEVLEV(lspec,i,j)] + C_tmp*abundance;
    }

  }


  /* Calculate the reverse (excitation) rate coefficients from detailed balance */
  /* i.e. C_ji = C_ij * g_i/g_j * exp( -(E_i-E_j)/ (kb T) )                     */

  for (ckr=0; ckr<ncoltran[LSPECPAR(lspec,par_max_ncoltran)]; ckr++){

    i = icol[LSPECPARTRAN(lspec,par_max_ncoltran,ckr)];
    j = jcol[LSPECPARTRAN(lspec,par_max_ncoltran,ckr)];

    C_coeff[LSPECLEVLEV(lspec,j,i)] = C_coeff[LSPECLEVLEV(lspec,i,j)]
                                     * weight[LSPECLEV(lspec,i)] / weight[LSPECLEV(lspec,j)]
                                     * exp( -( energy[LSPECLEV(lspec,i)]
                                               -energy[LSPECLEV(lspec,j)] )
                                             / (KB*temperature[gridp]) );

  }


  // printf("(calc_C_coeff): output [C_ij]= \n");

  // for (i=0; i<nlev[lspec]; i++){

  //   for (j=0; j<nlev[lspec]; j++){

  //     printf( "\t %.2lE", C_coeff[LSPECLEVLEV(lspec,i,j)] );
  //   }

  //   printf("\n");
  // }

}

/*-----------------------------------------------------------------------------------------------*/
