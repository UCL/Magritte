// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <string>

#include "declarations.hpp"
#include "reaction_rates.hpp"
#include "calc_reac_rates.hpp"
#include "calc_reac_rates_rad.hpp"



// reaction_rates: Check which kind of reaction and call appropriate rate calculator
// ---------------------------------------------------------------------------------

int reaction_rates (long ncells, CELL *cell, REACTIONS reactions, long o,
                    double *column_H2, double *column_HD, double *column_C, double *column_CO)
{

  // For all reactions

  for (int reac = 0; reac < NREAC; reac++)
  {

    // Copy reaction data to variables with more convenient names

    std::string R1 = reactions.R1[reac];   // reactant 1
    std::string R2 = reactions.R2[reac];   // reactant 2
    std::string R3 = reactions.R3[reac];   // reactant 3

    std::string P1 = reactions.P1[reac];   // reaction product 1
    std::string P2 = reactions.P2[reac];   // reaction product 2
    std::string P3 = reactions.P3[reac];   // reaction product 3
    std::string P4 = reactions.P4[reac];   // reaction product 4


    // The following rates are described in calc_reac_rates.c


    // H2 formation
    // Following Cazaux & Tielens (2002, ApJ, 575, L29) and (2004, ApJ, 604, 222)

    if (      R1 == "H"
         &&   R2 == "H"
         && ( R3 == "" || R3 == "#" )
         &&   P1 == "H2"
         && ( P2 == "" || P2 == "#" ) )
    {
      reactions.nr_H2_formation = reac;

      cell[o].rate[reac] = rate_H2_formation (cell, reactions, reac, o);
    }


    // Reactions involving PAHs
    // Following  Wolfire et al. (2003, ApJ, 587, 278; 2008, ApJ, 680, 384)

    else if (     R1 == "PAH+"  ||  R2 == "PAH+"  ||  R3 == "PAH+"
              ||  R1 == "PAH-"  ||  R2 == "PAH-"  ||  R3 == "PAH-"
              ||  R1 == "PAH0"  ||  R2 == "PAH0"  ||  R3 == "PAH0"
              ||  R1 == "PAH"   ||  R2 == "PAH"   ||  R3 == "PAH" )
    {
      cell[o].rate[reac] = rate_PAH (cell, reactions, reac, o);
    }


    // Cosmic ray induced ionization

    else if (R2 == "CRP")
    {
      cell[o].rate[reac] = rate_CRP (cell, reactions, reac, o);
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRAY")
    {
      cell[o].rate[reac] = 0.0;
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRSEC")
    {
      cell[o].rate[reac] = 0.0;
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRLYA")
    {
      cell[o].rate[reac] = 0.0;
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRPHOT")
    {
      cell[o].rate[reac] = 0.0;
    }


    // Photoreactions due to cosmic ray-induced secondary photons

    else if (R2 == "CRPHOT")
    {
      cell[o].rate[reac] = rate_CRPHOT (cell, reactions, reac, o);
    }


    // Freeze-out of neutral species

    else if (R2 == "FREEZE")
    {
      cell[o].rate[reac] = rate_FREEZE (cell, reactions, reac, o);
    }


    // Freeze-out of singly charged positive ions

    else if (R2 == "ELFRZE")
    {
      cell[o].rate[reac] = rate_ELFRZE (cell, reactions, reac, o);
    }


    // Desorption due to cosmic ray heating
    // Following Roberts et al. (2007, MNRAS, 382, 773, Equation 3)

    else if (R2 == "CRH")
    {
      cell[o].rate[reac] = rate_CRH (reactions, reac);
    }


    // Thermal desorption
    // Following Hasegawa, Herbst & Leung (1992, ApJS, 82, 167, Equations 2 & 3)

    else if (R2 == "THERM")
    {
      cell[o].rate[reac] = rate_THERM (cell, reactions, reac, o);
    }


    // Grain mantle reaction

    else if (R2 == "#")
    {
      cell[o].rate[reac] = rate_GM (reactions, reac);
    }





    // The following 5 rates are described in calc_reac_rates_rad.c


    // Photodesorption

    else if (R2 == "PHOTD")
    {
      cell[o].rate[reac] = rate_PHOTD (cell, reactions, reac, o);
    }


    // H2 photodissociation
    // Taking into account self-shielding and grain extinction

    else if ( (R1 == "H2")  &&  (R2 == "PHOTON")  &&  (R3 == "") )
    {
      reactions.nr_H2_photodissociation = reac;

      cell[o].rate[reac] = rate_H2_photodissociation (cell, reactions, reac, column_H2, o);
    }


    // HD photodissociation

    else if ( (R1 == "HD")  &&  (R2 == "PHOTON")  &&  (R3 == "") )
    {
      cell[o].rate[reac] = rate_H2_photodissociation (cell, reactions, reac, column_HD, o);
    }


    // CO photodissociation

    else if ( (R1 == "CO")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              && ( P1 == "C"  ||  P2 == "C"  ||  P3 == "C"  ||  P4 == "C")
              && ( P1 == "O"  ||  P2 == "O"  ||  P3 == "O"  ||  P4 == "O") )
    {
      cell[o].rate[reac] = rate_CO_photodissociation (cell, reactions, reac, column_CO, column_H2, o);
    }


    // C photoionization

    else if ( (R1 == "C")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              &&  ( (P1 == "C+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "C+") ) )
    {
      reactions.nr_C_ionization = reac;

      cell[o].rate[reac] = rate_C_photoionization (cell, reactions, reac, column_C, column_H2, o);
    }


    // SI photoionization

    else if ( (R1 == "S")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              &&  ( (P1 == "S+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "S+") ))
    {
      cell[o].rate[reac] = rate_SI_photoionization (cell, reactions, reac, o);
    }


    // Other (canonical) photoreactions

    else if (R2 == "PHOTON")
    {
      cell[o].rate[reac] = rate_canonical_photoreaction (cell, reactions, reac, o);
    }



    // The following reactions are again described in calc_reac_rates.cpp


    // All other reactions

    else
    {
      cell[o].rate[reac] = rate_canonical (cell, reactions, reac, o);
    }




    // Now all rates should be calculated


    // Check that the rate is physical (0<RATE(I)<1) and produce an error message if not.
    // Impose a lower cut-off on all rate coefficients to prevent the problem becoming too stiff
    // Rates less than 1E-99 are set to zero.
    // Grain-surface reactions and desorption mechanisms are allowed rates greater than 1.

    if (cell[o].rate[reac] < 0.0)
    {
      printf("(reaction_rates): ERROR, negative rate for reaction %d \n", reac);
    }

    else if ( (cell[o].rate[reac] > 1.0) && (R2 != "#") )
    {
      printf("(reaction_rates): WARNING, rate too large for reaction %d \n", reac);
      printf("(reaction_rates): WARNING, rate is set to 1.0 \n");

      cell[o].rate[reac] = 1.0;
    }

    else if (cell[o].rate[reac] < 1.0E-99)
    {
      cell[o].rate[reac] = 0.0;
    }


  } // end of reac loop over reactions


  return (0);

}
