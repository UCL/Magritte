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

int calc_C_coeff (long ncells, CELL *cell, LINE_SPECIES line_species,
                  double *C_coeff, long o, int lspec)
{

  // cell[0].temperature.gas = 100.0;
  // printf("TEMPTEMPTEMP %1.2lE\n", cell[0].temperature.gas);

  // Calculate H2 ortho/para fraction at equilibrium for given temperature

  double frac_H2_para  = 0.0;   // fraction of para-H2
  double frac_H2_ortho = 0.0;   // fraction of ortho-H2


  if (cell[o].abundance[nr_H2] > 0.0)
  {
    frac_H2_para  = 1.0 / (1.0 + 9.0*exp(-170.5/cell[o].temperature.gas));
    frac_H2_ortho = 1.0 - frac_H2_para;
  }


  // Initialize C_coeff

  initialize_double_array (TOT_NLEV2, C_coeff);


  // For all collision partners

  for (int par = 0; par < ncolpar[lspec]; par++)
  {

    // Get number of species corresponding to collision partner

    int spec = line_species.partner[LSPECPAR(lspec,par)];


    // Find available temperatures closest to actual tamperature

    int tindex_low  = -1;   // index of temperature below actual temperature
    int tindex_high = -1;   // index of temperature above actual temperature


    // Find data corresponding to temperatures above and below actual temperature
    // printf("\n");
    // printf("%d %d\n", lspec, par);
    // for (int tindex = 0; tindex < ncoltemp[LSPECPAR(lspec,par)]; tindex++)
    // {
    //   printf("%1.1lE ", line_species.coltemp[LSPECPARTEMP(lspec,par,tindex)]);
    // }
    // printf("\n");

    if (line_species.coltemp[LSPECPARTEMP(lspec,par,ncoltemp[LSPECPAR(lspec,par)]-1)] <= cell[o].temperature.gas)
    {
      tindex_high = tindex_low = ncoltemp[LSPECPAR(lspec,par)]-1;
    }

    else if (line_species.coltemp[LSPECPARTEMP(lspec,par,0)] >= cell[o].temperature.gas)
    {
      tindex_high = tindex_low = 0;
    }

    else
    {

      for (int tindex = 0; tindex < ncoltemp[LSPECPAR(lspec,par)]; tindex++)
      {
        // printf("coltemp %1.2lE\n", line_species.coltemp[LSPECPARTEMP(lspec,par,tindex)]);

        if (cell[o].temperature.gas < line_species.coltemp[LSPECPARTEMP(lspec,par,tindex)])
        {
          tindex_low  = tindex-1;
          tindex_high = tindex;

          break;
        }
      }

    }

    // printf("%d %d     tot %d\n", tindex_low, tindex_high, ncoltemp[LSPECPAR(lspec,par)]);


    double *C_T_low = new double[nlev[lspec]*nlev[lspec]];

    initialize_double_array (nlev[lspec]*nlev[lspec], C_T_low);

    double *C_T_high = new double[nlev[lspec]*nlev[lspec]];

    initialize_double_array (nlev[lspec]*nlev[lspec], C_T_high);


    for (int ckr = 0; ckr < ncoltran[LSPECPAR(lspec,par)]; ckr++)
    {
      int i = line_species.icol[LSPECPARTRAN(lspec,par,ckr)];
      int j = line_species.jcol[LSPECPARTRAN(lspec,par,ckr)];

      C_T_low[LINDEX(lspec,i,j)]  = line_species.C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_low)];
      C_T_high[LINDEX(lspec,i,j)] = line_species.C_data[LSPECPARTRANTEMP(lspec,par,ckr,tindex_high)];
    }


    // Calculate reverse (excitation) rate coefficients from detailed balance, if not given
    // i.e. C_ji = C_ij * g_i/g_j * exp( -(E_i-E_j)/ (kb T) )

    for (int ckr = 0; ckr < ncoltran[LSPECPAR(lspec,par)]; ckr++)
    {
      int i = line_species.icol[LSPECPARTRAN(lspec,par,ckr)];
      int j = line_species.jcol[LSPECPARTRAN(lspec,par,ckr)];

      int l_i = LSPECLEV(lspec,i);
      int l_j = LSPECLEV(lspec,j);

      if ( (C_T_low[LINDEX(lspec,j,i)] == 0.0) && (C_T_low[LINDEX(lspec,i,j)] != 0.0) )
      {
        C_T_low[LINDEX(lspec,j,i)] = C_T_low[LINDEX(lspec,i,j)] * line_species.weight[l_i]/line_species.weight[l_j]
                               * exp( -(line_species.energy[l_i] - line_species.energy[l_j])
                                       /(KB*line_species.coltemp[LSPECPARTEMP(lspec,par,tindex_low)]) );
      }

      if ( (C_T_high[LINDEX(lspec,j,i)] == 0.0) && (C_T_high[LINDEX(lspec,i,j)] != 0.0) )
      {
        C_T_high[LINDEX(lspec,j,i)] = C_T_high[LINDEX(lspec,i,j)] * line_species.weight[l_i]/line_species.weight[l_j]
                                * exp( -(line_species.energy[l_i] - line_species.energy[l_j])
                                        /(KB*line_species.coltemp[LSPECPARTEMP(lspec,par,tindex_high)]) );
      }
    }


    // Calculate (linear) interpolation step

    double step = 0.0;

    if (tindex_high != tindex_low)
    {
      step = (cell[o].temperature.gas - line_species.coltemp[LSPECPARTEMP(lspec,par,tindex_low)])
              / ( line_species.coltemp[LSPECPARTEMP(lspec,par,tindex_high)]
                  - line_species.coltemp[LSPECPARTEMP(lspec,par,tindex_low)] );
    }

        // printf ("T %1.2lE %1.2lE   %1.2lE\n", line_species.coltemp[LSPECPARTEMP(lspec,par,tindex_low)],
                                              // line_species.coltemp[LSPECPARTEMP(lspec,par,tindex_high)], step );




    // Weigh contributions to C by abundance

    double abundance = cell[o].density * cell[o].abundance[spec];


    if      (line_species.ortho_para[LSPECPAR(lspec,par)] == 'o')
    {
      // printf("O\n");
      abundance = abundance * frac_H2_ortho;
    }

    else if (line_species.ortho_para[LSPECPAR(lspec,par)] == 'p')
    {
      // printf("P\n");
      abundance = abundance * frac_H2_para;
    }

    // printf("abn %1.2lE  spec %d \n", abundance, spec);


    // For all C matrix elements

    for (int i = 0; i < nlev[lspec]; i++)
    {
      for (int j = 0; j < nlev[lspec]; j++)
      {

        // Make a linear interpolation for C in temperature

        double C_tmp = C_T_low[LINDEX(lspec,i,j)] + (C_T_high[LINDEX(lspec,i,j)] - C_T_low[LINDEX(lspec,i,j)]) * step;

        C_coeff[LSPECLEVLEV(lspec,i,j)] = C_coeff[LSPECLEVLEV(lspec,i,j)] + C_tmp*abundance;
      }
    }


    delete [] C_T_low;
    delete [] C_T_high;

  } // end of par loop over collision partners


  return(0);

}
