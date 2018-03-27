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

int reaction_rates (long ncells, CELLS *cells, REACTIONS reactions, long o,
                    double *column_H2, double *column_HD, double *column_C, double *column_CO)
{
  printf("OK\n");
  // For all reactions



  for (int e = 0; e < NREAC; e++)
  {

    // Copy reaction data to variables with more convenient names

    std::string R1 = reactions.R1[e];   // reactant 1
    std::string R2 = reactions.R2[e];   // reactant 2
    std::string R3 = reactions.R3[e];   // reactant 3

    std::string P1 = reactions.P1[e];   // reaction product 1
    std::string P2 = reactions.P2[e];   // reaction product 2
    std::string P3 = reactions.P3[e];   // reaction product 3
    std::string P4 = reactions.P4[e];   // reaction product 4


    // The following rates are described in calc_reac_rates.c


    // H2 formation
    // Following Cazaux & Tielens (2002, ApJ, 575, L29) and (2004, ApJ, 604, 222)

    if (      R1 == "H"
         &&   R2 == "H"
         && ( R3 == "" || R3 == "#" )
         &&   P1 == "H2"
         && ( P2 == "" || P2 == "#" ) )
    {
      reactions.nr_H2_formation = e;

      cells->rate[READEX(o,e)] = rate_H2_formation (cells, reactions, e, o);
    }


    // Reactions involving PAHs
    // Following  Wolfire et al. (2003, ApJ, 587, 278; 2008, ApJ, 680, 384)

    else if (     R1 == "PAH+"  ||  R2 == "PAH+"  ||  R3 == "PAH+"
              ||  R1 == "PAH-"  ||  R2 == "PAH-"  ||  R3 == "PAH-"
              ||  R1 == "PAH0"  ||  R2 == "PAH0"  ||  R3 == "PAH0"
              ||  R1 == "PAH"   ||  R2 == "PAH"   ||  R3 == "PAH" )
    {
      cells->rate[READEX(o,e)] = rate_PAH (cells, reactions, e, o);
    }


    // Cosmic ray induced ionization

    else if (R2 == "CRP")
    {
      cells->rate[READEX(o,e)] = rate_CRP (cells, reactions, e, o);
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRAY")
    {
      cells->rate[READEX(o,e)] = 0.0;
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRSEC")
    {
      cells->rate[READEX(o,e)] = 0.0;
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRLYA")
    {
      cells->rate[READEX(o,e)] = 0.0;
    }


    // X-ray induced reactions
    // Not yet included in 3D-PDR...

    else if (R2 == "XRPHOT")
    {
      cells->rate[READEX(o,e)] = 0.0;
    }


    // Photoreactions due to cosmic ray-induced secondary photons

    else if (R2 == "CRPHOT")
    {
      cells->rate[READEX(o,e)] = rate_CRPHOT (cells, reactions, e, o);
    }


    // Freeze-out of neutral species

    else if (R2 == "FREEZE")
    {
      cells->rate[READEX(o,e)] = rate_FREEZE (cells, reactions, e, o);
    }


    // Freeze-out of singly charged positive ions

    else if (R2 == "ELFRZE")
    {
      cells->rate[READEX(o,e)] = rate_ELFRZE (cells, reactions, e, o);
    }


    // Desorption due to cosmic ray heating
    // Following Roberts et al. (2007, MNRAS, 382, 773, Equation 3)

    else if (R2 == "CRH")
    {
      cells->rate[READEX(o,e)] = rate_CRH (reactions, e);
    }


    // Thermal desorption
    // Following Hasegawa, Herbst & Leung (1992, ApJS, 82, 167, Equations 2 & 3)

    else if (R2 == "THERM")
    {
      cells->rate[READEX(o,e)] = rate_THERM (cells, reactions, e, o);
    }


    // Grain mantle reaction

    else if (R2 == "#")
    {
      cells->rate[READEX(o,e)] = rate_GM (reactions, e);
    }





    // The following 5 rates are described in calc_reac_rates_rad.c


    // Photodesorption

    else if (R2 == "PHOTD")
    {
      cells->rate[READEX(o,e)] = rate_PHOTD (cells, reactions, e, o);
    }


    // H2 photodissociation
    // Taking into account self-shielding and grain extinction

    else if ( (R1 == "H2")  &&  (R2 == "PHOTON")  &&  (R3 == "") )
    {
      reactions.nr_H2_photodissociation = e;

      cells->rate[READEX(o,e)] = rate_H2_photodissociation (cells, reactions, e, column_H2, o);
    }


    // HD photodissociation

    else if ( (R1 == "HD")  &&  (R2 == "PHOTON")  &&  (R3 == "") )
    {
      cells->rate[READEX(o,e)] = rate_H2_photodissociation (cells, reactions, e, column_HD, o);
    }


    // CO photodissociation

    else if ( (R1 == "CO")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              && ( P1 == "C"  ||  P2 == "C"  ||  P3 == "C"  ||  P4 == "C")
              && ( P1 == "O"  ||  P2 == "O"  ||  P3 == "O"  ||  P4 == "O") )
    {
      cells->rate[READEX(o,e)] = rate_CO_photodissociation (cells, reactions, e, column_CO, column_H2, o);
    }


    // C photoionization

    else if ( (R1 == "C")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              &&  ( (P1 == "C+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "C+") ) )
    {
      reactions.nr_C_ionization = e;

      cells->rate[READEX(o,e)] = rate_C_photoionization (cells, reactions, e, column_C, column_H2, o);
    }


    // SI photoionization

    else if ( (R1 == "S")  &&  (R2 == "PHOTON")  &&  (R3 == "")
              &&  ( (P1 == "S+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "S+") ))
    {
      cells->rate[READEX(o,e)] = rate_SI_photoionization (cells, reactions, e, o);
    }


    // Other (canonical) photoreactions

    else if (R2 == "PHOTON")
    {
      cells->rate[READEX(o,e)] = rate_canonical_photoreaction (cells, reactions, e, o);
    }



    // The following reactions are again described in calc_reac_rates.cpp


    // All other reactions

    else
    {
      cells->rate[READEX(o,e)] = rate_canonical (cells, reactions, e, o);
    }




    // Now all rates should be calculated


    // Check that the rate is physical (0<RATE(I)<1) and produce an error message if not.
    // Impose a lower cut-off on all rate coefficients to prevent the problem becoming too stiff
    // Rates less than 1E-99 are set to zero.
    // Grain-surface reactions and desorption mechanisms are allowed rates greater than 1.

    if (cells->rate[READEX(o,e)] < 0.0)
    {
      printf("(reaction_rates): ERROR, negative rate for reaction %d \n", e);
    }

    else if ( (cells->rate[READEX(o,e)] > 1.0) && (R2 != "#") )
    {
      printf("(reaction_rates): WARNING, rate too large for reaction %d \n", e);
      printf("(reaction_rates): WARNING, rate is set to 1.0 \n");

      cells->rate[READEX(o,e)] = 1.0;
    }

    else if (cells->rate[READEX(o,e)] < 1.0E-99)
    {
      cells->rate[READEX(o,e)] = 0.0;
    }


  } // end of reac loop over reactions


  return (0);

}
