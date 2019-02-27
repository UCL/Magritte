// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "linedata.hpp"


///  Constructor for LINEDATA
/////////////////////////////

LINEDATA ::
    LINEDATA                 (
        const int nlev_local,
        const int nrad_local )
 : nlev       (nlev_local)
 , nrad       (nrad_local)
 , irad       (nrad)
 , jrad       (nrad)
 , energy     (nlev)
 , weight     (nlev)
 , population (nlev)
 , frequency  (nrad)
 , emissivity (nrad)
 , opacity    (nrad)
 , A    (nlev, nlev)
 , B    (nlev, nlev)
{


}   // END OF CONSTRUCTOR




///  compute_populations_LTE
////////////////////////////

void LINEDATA ::
    compute_populations_LTE      (
        const double temperature )
{

  // Calculate fractional LTE level populations and partition function

  double partition_function = 0.0;

  for (int i = 0; i < nlev; i++)
  {
    population(i) = weight(i) * exp(-energy(i)/(KB*temperature));

    partition_function += population(i);
  }


  // Rescale (normalize) LTE level populations

  const double inverse_partition_function = 1.0 / partition_function;

  for (int i = 0; i < nlev; i++)
  {
    population(i) *= inverse_partition_function;
  }


}


///  compute_emissivity_and_opacity:
////////////////////////////////////

void LINEDATA ::
    compute_emissivity_and_opacity (void)
{

  for (int k = 0; k < nrad; k++)
  {
    const int i = irad[k];
    const int j = jrad[k];

    const double hv_4pi = HH_OVER_FOUR_PI * frequency(k);

    emissivity(k) = hv_4pi * A(i,j) * population(i);
       opacity(k) = hv_4pi * (B(j,i) * population(j) - B(i,j) * population(i));
  }

}
