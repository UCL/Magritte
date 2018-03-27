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

double cooling (long ncells, CELLS *cells, LINES lines, long o)
{

  double cooling_radiative = 0.0;   // radiative cooling


  // For all line producing species

  for (int ls = 0; ls < NLSPEC; ls++)
  {

    // For all transitions with (i > j)

    for (int kr = 0; kr < nrad[ls]; kr++)
    {

      long m_ij = LSPECRAD(ls,kr);   // mean intensity index

      int i = lines.irad[m_ij];   // i level index corresponding to transition kr
      int j = lines.jrad[m_ij];   // j level index corresponding to transition kr

      long b_ij = LSPECLEVLEV(ls,i,j);         // A_coeff, B_coeff and frequency index
      long b_ji = LSPECLEVLEV(ls,j,i);         // A_coeff, B_coeff and frequency index

      long p_i  = LINDEX(o,LSPECLEV(ls,i));   // population at level i
      long p_j  = LINDEX(o,LSPECLEV(ls,j));   // population at level j


      // Calculate source function

      double Source = 0.0;

      double factor = 2.0 * HH * pow(lines.frequency[b_ij], 3) / pow(CC, 2);

      double tpop   =   cells->pop[p_j] * lines.weight[LSPECLEV(ls,i)]
                      / cells->pop[p_i] / lines.weight[LSPECLEV(ls,j)] - 1.0;


      if (fabs(tpop) < 1.0E-50)
      {
        Source = HH * lines.frequency[b_ij] * cells->pop[p_i] * lines.A_coeff[b_ij] / 4.0 / PI;
      }

      else if (cells->pop[p_i] > 0.0)
      {
        Source = factor / tpop;
      }


      // Calculate radiative line cooling

      if (Source != 0.0)
      {
        cooling_radiative = cooling_radiative
                            + HH*lines.frequency[b_ij] * lines.A_coeff[b_ij] * cells->pop[p_i]
                              * (1.0 - cells->mean_intensity[KINDEX(o,m_ij)]/Source);
      }


    } // end of kr loop over transitions

  } // end of lspec loop over line producing species


  double cooling_total = cooling_radiative;


  return cooling_total;

}
