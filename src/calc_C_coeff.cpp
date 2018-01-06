// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_C_coeff.hpp"
#include "initializers.hpp"


// calc_C_coeff: calculates collisional coefficients (C_ij) from line data
// -----------------------------------------------------------------------

int calc_C_coeff (long ncells, CELL *cell, double *C_data, double *coltemp, int *icol, int *jcol,
                  double *weight, double *energy, double *C_coeff, long gridp, int lspec)
{

  // Calculate H2 ortho/para fraction at equilibrium for given temperature

  double frac_H2_para  = 0.0;   // fraction of para-H2
  double frac_H2_ortho = 0.0;   // fraction of ortho-H2


  if (cell[gridp].abundance[H2_nr] > 0.0)
  {
    frac_H2_para  = 1.0 / (1.0 + 9.0*exp(-170.5/cell[gridp].temperature.gas));
    frac_H2_ortho = 1.0 - frac_H2_para;
  }

  // printf("fsdsl;kjsjkid\n");


  // Initialize C_coeff

  initialize_double_array(C_coeff, TOT_NLEV2);

  // printf("yrdfiojdfiojdfsijofvidojfvid\n");

  // For all collision partners

  for (int par = 0; par < ncolpar[lspec]; par++)
  {

    // Get the number of the species corresponding to the collision partner

    int spec = spec_par[lspec,par];


    // Find the available temperatures closest to the actual tamperature

    int tindex_low  = -1;   // index of temperature below actual temperature
    int tindex_high = -1;   // index of temperature above actual temperature


    // Find data corresponding to temperatures above and below actual temperature

    for (int tindex = 0; tindex < ncoltemp[LSPECPAR(lspec,par)]; tindex++)
    {
      if (cell[gridp].temperature.gas < coltemp[LSPECPARTEMP(lspec,par,tindex)])
      {
        tindex_low  = tindex-1;
        tindex_high = tindex;

        break;
      }
    }

    if (tindex_high == -1)
    {
      tindex_high = tindex_low = ncoltemp[LSPECPAR(lspec,par)]-1;
    }

    if (tindex_high == 0)
    {
      tindex_high = tindex_low = 0;
    }


    double *C_T_low;
    C_T_low = (double*) malloc( nlev[lspec]*nlev[lspec]*sizeof(double) );

    initialize_double_array(C_T_low, nlev[lspec]*nlev[lspec]);

    double *C_T_high;
    C_T_high = (double*) malloc( nlev[lspec]*nlev[lspec]*sizeof(double) );

    initialize_double_array(C_T_high, nlev[lspec]*nlev[lspec]);


    for (int ckr = 0; ckr < ncoltran[LSPECPAR(lspec,par)]; ckr++)
    {
      int i = icol[LSPECPARTRAN(lspec,par,ckr)];
      int j = jcol[LSPECPARTRAN(lspec,par,ckr)];

      C_T_low[LINDEX(i,j)]  = C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_low)];
      C_T_high[LINDEX(i,j)] = C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_high)];
    }


    // Calculate reverse (excitation) rate coefficients from detailed balance, if not given
    // i.e. C_ji = C_ij * g_i/g_j * exp( -(E_i-E_j)/ (kb T) )

    for (int ckr = 0; ckr < ncoltran[LSPECPAR(lspec,par)]; ckr++)
    {
      int i = icol[LSPECPARTRAN(lspec,par,ckr)];
      int j = jcol[LSPECPARTRAN(lspec,par,ckr)];

      int l_i = LSPECLEV(lspec,i);
      int l_j = LSPECLEV(lspec,j);

      if ( (C_T_low[LINDEX(j,i)] == 0.0) && (C_T_low[LINDEX(i,j)] != 0.0) )
      {
        C_T_low[LINDEX(j,i)] = C_T_low[LINDEX(i,j)] * weight[l_i]/weight[l_j]
                               * exp( -(energy[l_i] - energy[l_j])
                                       /(KB*coltemp[LSPECPARTEMP(lspec,par,tindex_low)]) );
      }

      if ( (C_T_high[LINDEX(j,i)] == 0.0) && (C_T_high[LINDEX(i,j)] != 0.0) )
      {
        C_T_high[LINDEX(j,i)] = C_T_high[LINDEX(i,j)] * weight[l_i]/weight[l_j]
                                * exp( -(energy[l_i] - energy[l_j])
                                        /(KB*coltemp[LSPECPARTEMP(lspec,par,tindex_high)]) );
      }
    }


    // Calculate the (linear) interpolation step

    double step = 0.0;

    if (tindex_high != tindex_low)
    {
      step = (cell[gridp].temperature.gas - coltemp[LSPECPARTEMP(lspec,par,tindex_low)])
              / ( coltemp[LSPECPARTEMP(lspec,par,tindex_high)]
                  - coltemp[LSPECPARTEMP(lspec,par,tindex_low)] );
    }


    // For all C matrix elements

    for (int i = 0; i < nlev[lspec]; i++)
    {
      for (int j = 0; j < nlev[lspec]; j++)
      {

        // Make a linear interpolation for C in temperature

        double C_tmp = C_T_low[LINDEX(i,j)]
                       + (C_T_high[LINDEX(i,j)] - C_T_low[LINDEX(i,j)]) * step;


        // Weigh contributions to C by abundance

        double abundance = cell[gridp].density * cell[gridp].abundance[spec];

        if      (ortho_para[LSPECPAR(lspec,par)] == 'o')
        {
          abundance = abundance * frac_H2_ortho;
        }

        else if (ortho_para[LSPECPAR(lspec,par)] == 'p')
        {
          abundance = abundance * frac_H2_para;
        }

        C_coeff[LSPECLEVLEV(lspec,i,j)] = C_coeff[LSPECLEVLEV(lspec,i,j)] + C_tmp*abundance;

      }
    }


    free(C_T_low);
    free(C_T_high);

  } // end of par loop over collision partners


  return(0);

}
