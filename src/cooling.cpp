// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>

#include "declarations.hpp"
#include "cooling.hpp"



// cooling: calculate total cooling
// --------------------------------

double cooling (long ncells, LINE_SPECIES line_species, long gridp, double *pop, double *mean_intensity)
{

  double cooling_radiative = 0.0;   // radiative cooling


  // For all line producing species

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {

    // For all transitions with (i > j)

    for (int kr = 0; kr < nrad[lspec]; kr++)
    {

      int i = line_species.irad[LSPECRAD(lspec,kr)];   // i level index corresponding to transition kr
      int j = line_species.jrad[LSPECRAD(lspec,kr)];   // j level index corresponding to transition kr

      long b_ij = LSPECLEVLEV(lspec,i,j);         // A_coeff, B_coeff and frequency index
      long b_ji = LSPECLEVLEV(lspec,j,i);         // A_coeff, B_coeff and frequency index

      long p_i  = LSPECGRIDLEV(lspec,gridp,i);    // population at level i
      long p_j  = LSPECGRIDLEV(lspec,gridp,j);    // population at level j

      long m_ij = LSPECGRIDRAD(lspec,gridp,kr);   // mean intensity index


      // Calculate source function

      double Source = 0.0;

      double factor = 2.0 * HH * pow(line_species.frequency[b_ij], 3) / pow(CC, 2);

      double tpop   =   pop[p_j] * line_species.weight[LSPECLEV(lspec,i)]
                      / pop[p_i] / line_species.weight[LSPECLEV(lspec,j)] - 1.0;


      if (fabs(tpop) < 1.0E-50)
      {
        Source = HH * line_species.frequency[b_ij] * pop[p_i] * line_species.A_coeff[b_ij] / 4.0 / PI;
      }

      else if (pop[p_i] > 0.0)
      {
        Source = factor / tpop;
      }


      // Calculate radiative line cooling

      if (Source != 0.0)
      {
        cooling_radiative = cooling_radiative
                            + HH*line_species.frequency[b_ij] * line_species.A_coeff[b_ij] * pop[p_i]
                              * (1.0 - mean_intensity[m_ij]/Source);
      }


    } // end of kr loop over transitions

  } // end of lspec loop over line producing species


  double cooling_total = cooling_radiative;


  return cooling_total;

}
