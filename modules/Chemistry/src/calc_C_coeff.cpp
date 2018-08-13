// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "declarations.hpp"
#include "calc_C_coeff.hpp"
#include "initializers.hpp"


// calc_C_coeff: calculates collisional coefficients (C_ij) from line data
// -----------------------------------------------------------------------

int calc_C_coeff (CELLS *cells, SPECIES species, LINES lines,
                  double *C_coeff, long o, int ls)
{

  // Calculate H2 ortho/para fraction at equilibrium for given temperature

  double frac_H2_para  = 0.0;   // fraction of para-H2
  double frac_H2_ortho = 0.0;   // fraction of ortho-H2


  if (cells->abundance[SINDEX(o,species.nr_H2)] > 0.0)
  {
    frac_H2_para  = 1.0 / (1.0 + 9.0*exp(-170.5/cells->temperature_gas[o]));
    frac_H2_ortho = 1.0 - frac_H2_para;
  }


  // Initialize C_coeff

  initialize_double_array (TOT_NLEV2, C_coeff);


  // For all collision partners

  for (int par = 0; par < ncolpar[ls]; par++)
  {

    // Get number of species corresponding to collision partner

    int spec = lines.partner[LSPECPAR(ls,par)];


    // Find available temperatures closest to actual tamperature

    int tindex_low  = -1;   // index of temperature below actual temperature
    int tindex_high = -1;   // index of temperature above actual temperature


    // Find data corresponding to temperatures above and below actual temperature

    if (lines.coltemp[LSPECPARTEMP(ls,par,ncoltemp[LSPECPAR(ls,par)]-1)] <= cells->temperature_gas[o])
    {
      tindex_high = tindex_low = ncoltemp[LSPECPAR(ls,par)]-1;
    }

    else if (lines.coltemp[LSPECPARTEMP(ls,par,0)] >= cells->temperature_gas[o])
    {
      tindex_high = tindex_low = 0;
    }

    else
    {

      for (int tindex = 0; tindex < ncoltemp[LSPECPAR(ls,par)]; tindex++)
      {
        if (cells->temperature_gas[o] < lines.coltemp[LSPECPARTEMP(ls,par,tindex)])
        {
          tindex_low  = tindex-1;
          tindex_high = tindex;

          break;
        }
      }

    }


    double *C_T_low = new double[nlev[ls]*nlev[ls]];

    initialize_double_array (nlev[ls]*nlev[ls], C_T_low);

    double *C_T_high = new double[nlev[ls]*nlev[ls]];

    initialize_double_array (nlev[ls]*nlev[ls], C_T_high);


    for (int ckr = 0; ckr < ncoltran[LSPECPAR(ls,par)]; ckr++)
    {
      int i = lines.icol[LSPECPARTRAN(ls,par,ckr)];
      int j = lines.jcol[LSPECPARTRAN(ls,par,ckr)];

      C_T_low[LLINDEX(ls,i,j)]  = lines.C_data[LSPECPARTRANTEMP(ls,par,ckr,tindex_low)];
      C_T_high[LLINDEX(ls,i,j)] = lines.C_data[LSPECPARTRANTEMP(ls,par,ckr,tindex_high)];
    }


    // Calculate reverse (excitation) rate coefficients from detailed balance, if not given
    // i.e. C_ji = C_ij * g_i/g_j * exp( -(E_i-E_j) / (kb T) )

    for (int ckr = 0; ckr < ncoltran[LSPECPAR(ls,par)]; ckr++)
    {
      int i = lines.icol[LSPECPARTRAN(ls,par,ckr)];
      int j = lines.jcol[LSPECPARTRAN(ls,par,ckr)];

      int l_i = LSPECLEV(ls,i);
      int l_j = LSPECLEV(ls,j);

      if ( (C_T_low[LLINDEX(ls,j,i)] == 0.0) && (C_T_low[LLINDEX(ls,i,j)] != 0.0) )
      {
        C_T_low[LLINDEX(ls,j,i)] = C_T_low[LLINDEX(ls,i,j)] * lines.weight[l_i]/lines.weight[l_j]
                               * exp( -(lines.energy[l_i] - lines.energy[l_j])
                                       /(KB*lines.coltemp[LSPECPARTEMP(ls,par,tindex_low)]) );
      }

      if ( (C_T_high[LLINDEX(ls,j,i)] == 0.0) && (C_T_high[LLINDEX(ls,i,j)] != 0.0) )
      {
        C_T_high[LLINDEX(ls,j,i)] = C_T_high[LLINDEX(ls,i,j)] * lines.weight[l_i]/lines.weight[l_j]
                                * exp( -(lines.energy[l_i] - lines.energy[l_j])
                                        /(KB*lines.coltemp[LSPECPARTEMP(ls,par,tindex_high)]) );
      }
    }


    // Calculate (linear) interpolation step

    double step = 0.0;

    if (tindex_high != tindex_low)
    {
      step = (cells->temperature_gas[o] - lines.coltemp[LSPECPARTEMP(ls,par,tindex_low)])
              / ( lines.coltemp[LSPECPARTEMP(ls,par,tindex_high)]
                  - lines.coltemp[LSPECPARTEMP(ls,par,tindex_low)] );
    }


    // Weigh contributions to C by abundance

    double abundance = cells->density[o] * cells->abundance[SINDEX(o,spec)];


    if      (lines.ortho_para[LSPECPAR(ls,par)] == 'o')
    {
      abundance = abundance * frac_H2_ortho;
    }

    else if (lines.ortho_para[LSPECPAR(ls,par)] == 'p')
    {
      abundance = abundance * frac_H2_para;
    }


    // Make a linear interpolation for C in temperature

    for (int i = 0; i < nlev[ls]; i++)
    {
      for (int j = 0; j < nlev[ls]; j++)
      {
        long s_ij  = LLINDEX(ls,i,j);
        long ss_ij = LSPECLEVLEV(ls,i,j);

        double C_tmp = C_T_low[s_ij] + (C_T_high[s_ij] - C_T_low[s_ij]) * step;

        C_coeff[ss_ij] = C_coeff[ss_ij] + C_tmp*abundance;
      }
    }


    delete [] C_T_low;
    delete [] C_T_high;

  } // end of par loop over collision partners


  return(0);

}
