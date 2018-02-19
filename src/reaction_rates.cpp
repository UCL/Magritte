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

int reaction_rates (long ncells, CELL *cell, REACTION *reaction, long o,
                    double *column_H2, double *column_HD, double *column_C, double *column_CO)
{

  // For all reactions

  for (int reac = 0; reac < NREAC; reac++)
  {

    // Copy reaction data to variables with more convenient names

    std::string R1 = reaction[reac].R1;   // reactant 1
    std::string R2 = reaction[reac].R2;   // reactant 2
    std::string R3 = reaction[reac].R3;   // reactant 3

    std::string P1 = reaction[reac].P1;   // reaction product 1
    std::string P2 = reaction[reac].P2;   // reaction product 2
    std::string P3 = reaction[reac].P3;   // reaction product 3
    std::string P4 = reaction[reac].P4;   // reaction product 4


    // The following rates are described in calc_reac_rates.c


    // H2 formation
    // Following Cazaux & Tielens (2002, ApJ, 575, L29) and (2004, ApJ, 604, 222)

    if (      R1 == "H"
         &&   R2 == "H"
         && ( R3 == "" || R3 == "#" )
         &&   P1 == "H2"
         && ( P2 == "" || P2 == "#" ) )
    {
      nr_H2_formation = reac;

      cell[o].rate[reac] = rate_H2_formation (cell, reaction, reac, o);
    }


    // Reactions involving PAHs
    // Following  Wolfire et al. (2003, ApJ, 587, 278; 2008, ApJ, 680, 384)

    else if (     R1 == "PAH+"  ||  R2 == "PAH+"  ||  R3 == "PAH+"
              ||  R1 == "PAH-"  ||  R2 == "PAH-"  ||  R3 == "PAH-"
              ||  R1 == "PAH0"  ||  R2 == "PAH0"  ||  R3 == "PAH0"
              ||  R1 == "PAH"   ||  R2 == "PAH"   ||  R3 == "PAH" )
    {
      cell[o].rate[reac] = rate_PAH (cell, reaction, reac, o);
    }


    // Cosmic ray induced ionization

    else if (R2 == "CRP")
    {
      cell[o].rate[reac] = rate_CRP (cell, reaction, reac, o);
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
      cell[o].rate[reac] = rate_CRPHOT (cell, reaction, reac, o);
    }


    // Freeze-out of neutral species

    else if (R2 == "FREEZE")
    {
      cell[o].rate[reac] = rate_FREEZE (cell, reaction, reac, o);
    }


    // Freeze-out of singly charged positive ions

    else if (R2 == "ELFRZE")
    {
      cell[o].rate[reac] = rate_ELFRZE (cell, reaction, reac, o);
    }


    // Desorption due to cosmic ray heating
    // Following Roberts et al. (2007, MNRAS, 382, 773, Equation 3)

    else if (R2 == "CRH")
    {
      cell[o].rate[reac] = rate_CRH (reaction, reac);
    }


    // Thermal desorption
    // Following Hasegawa, Herbst & Leung (1992, ApJS, 82, 167, Equations 2 & 3)

    else if (R2 == "THERM")
    {
      cell[o].rate[reac] = rate_THERM (cell, reaction, reac, o);
    }


    // Grain mantle reaction

    else if (R2 == "#")
    {
      cell[o].rate[reac] = rate_GM (reaction, reac);
    }





    // The following 5 rates are described in calc_reac_rates_rad.c


    // Photodesorption

    else if (R2 == "PHOTD")
    {
      cell[o].rate[reac] = rate_PHOTD (cell, reaction, reac, o);
    }


    // H2 photodissociation
    // Taking into account self-shielding and grain extinction

    else if ( (R1 == "H2")  &&  (R2 == "PHOTON")  &&  (R3 == "") )
    {
      nr_H2_photodissociation = reac;

      cell[o].rate[reac] = rate_H2_photodissociation (cell, reaction, reac, column_H2, o);
    }


    // HD photodissociation

    else if ( (R1 == "HD")  &&  (R2 == "PHOTON")  &&  (R3 == "") )
    {
      cell[o].rate[reac] = rate_H2_photodissociation (cell, reaction, reac, column_HD, o);
    }


    // CO photodissociation

    else if ( (R1 == "CO")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              && ( P1 == "C"  ||  P2 == "C"  ||  P3 == "C"  ||  P4 == "C")
              && ( P1 == "O"  ||  P2 == "O"  ||  P3 == "O"  ||  P4 == "O") )
    {
      cell[o].rate[reac] = rate_CO_photodissociation (cell, reaction, reac, column_CO, column_H2, o);
    }


    // C photoionization

    else if ( (R1 == "C")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              &&  ( (P1 == "C+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "C+") ) )
    {
      nr_C_ionization = reac;

      cell[o].rate[reac] = rate_C_photoionization (cell, reaction, reac, column_C, column_H2, o);
    }


    // SI photoionization

    else if ( (R1 == "S")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              &&  ( (P1 == "S+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "S+") ))
    {
      cell[o].rate[reac] = rate_SI_photoionization (cell, reaction, reac, o);
    }


    // Other (canonical) photoreaction

    else if (R2 == "PHOTON")
    {
      cell[o].rate[reac] = rate_canonical_photoreaction (cell, reaction, reac, o);
    }



    // The following reactions are again described in calc_reac_rates.cpp


    // All other reactions

    else
    {
      cell[o].rate[reac] = rate_canonical (cell, reaction, reac, o);
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
