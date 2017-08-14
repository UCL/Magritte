/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* reaction_rates: Check which reaction and calculate the reaction rate coefficient (k)          */
/*                                                                                               */
/* (based on calc_reac_rates in 3D-PDR)                                                          */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <string>
#include <string.h>
#include <math.h>

#include "declarations.hpp"
#include "reaction_rates.hpp"
#include "rate_calculations.hpp"
#include "rate_calculations_radfield.hpp"



/* reaction_rates: Check which kind of reaction and call appropriate rate calculator b           */
/*-----------------------------------------------------------------------------------------------*/

void reaction_rates( double *temperature_gas, double *temperature_dust,
                     double *rad_surface, double *AV,
                     double *column_H2, double *column_HD, double *column_C, double *column_CO,
                     double v_turb, long gridp )
{


  int reac, rc;                                                                /* reaction index */

  string R1;                                                                       /* reactant 1 */
  string R2;                                                                       /* reactant 2 */
  string R3;                                                                       /* reactant 3 */

  string P1;                                                               /* reaction product 1 */
  string P2;                                                               /* reaction product 2 */
  string P3;                                                               /* reaction product 3 */
  string P4;                                                               /* reaction product 4 */

  /* All rate functions can be found in rate_calculations.cpp and rate_calculations_radfield.cpp */
  /* The rate functions are calculated locally so only need the

  /* For all reactions */

  for (reac=0; reac<NREAC; reac++){


    /* Copy the reaction data to variables with more convenient names */

    R1 = reaction[reac].R1;
    R2 = reaction[reac].R2;
    R3 = reaction[reac].R3;

    P1 = reaction[reac].P1;
    P2 = reaction[reac].P2;
    P3 = reaction[reac].P3;
    P4 = reaction[reac].P4;




    /* The following rates are described in rate_calculations.c


    /* H2 formation */
    /* Following Cazaux & Tielens (2002, ApJ, 575, L29) and (2004, ApJ, 604, 222) */

    if (      R1 == "H"
         &&   R2 == "H"
         && ( R3 == "" || R3 == "#" )
         &&   P1 == "H2"
         && ( P2 == "" || P2 == "#" ) ){

      H2_formation_nr = reac;

      reaction[reac].k[gridp] = rate_H2_formation( reac, temperature_gas[gridp],
                                                         temperature_dust[gridp] );
    }


    /* Reactions involving PAHs */
    /* Following  Wolfire et al. (2003, ApJ, 587, 278; 2008, ApJ, 680, 384) */

    else if ( R1 == "PAH+"  ||  R2 == "PAH+"  ||  R3 == "PAH+"
              ||  R1 == "PAH-"  ||  R2 == "PAH-"  ||  R3 == "PAH-"
              ||  R1 == "PAH0"  ||  R2 == "PAH0"  ||  R3 == "PAH0"
              ||  R1 == "PAH"   ||  R2 == "PAH"   ||  R3 == "PAH" ){

      reaction[reac].k[gridp] = rate_PAH(reac, temperature_gas[gridp]);
    }


    /* Cosmic ray induced ionization */

    else if ( R2 == "CRP" ){

      reaction[reac].k[gridp] = rate_CRP(reac, temperature_gas[gridp]);
    }


    /* X-ray induced reactions */
    /* Not yet included in 3D-PDR... */

    else if ( R2 == "XRAY" ){

      reaction[reac].k[gridp] = 0.0;
    }


    /* X-ray induced reactions */
    /* Not yet included in 3D-PDR... */

    else if ( R2 == "XRSEC" ){

      reaction[reac].k[gridp] = 0.0;
    }


    /* X-ray induced reactions */
    /* Not yet included in 3D-PDR... */

    else if ( R2 == "XRLYA" ){

      reaction[reac].k[gridp] = 0.0;
    }


    /* X-ray induced reactions */
    /* Not yet included in 3D-PDR... */

    else if ( R2 == "XRPHOT" ){

      reaction[reac].k[gridp] = 0.0;
    }


    /* Photoreactions due to cosmic ray-induced secondary photons */

    else if ( R2 == "CRPHOT" ){

      reaction[reac].k[gridp] = rate_CRPHOT(reac, temperature_gas[gridp]);
    }


    /* Freeze-out of neutral species */

    else if ( R2 == "FREEZE" ){

      reaction[reac].k[gridp] = rate_FREEZE(reac, temperature_gas[gridp]);
    }


    /* Freeze-out of singly charged positive ions */

    else if ( R2 == "ELFRZE" ){

      reaction[reac].k[gridp] = rate_ELFRZE(reac, temperature_gas[gridp]);
    }


    /* Desorption due to cosmic ray heating */
    /* Following Roberts et al. (2007, MNRAS, 382, 773, Equation 3) */

    else if ( R2 == "CRH" ){

      reaction[reac].k[gridp] = rate_CRH(reac, temperature_gas[gridp]);
    }


    /* Thermal desorption */
    /* Following Hasegawa, Herbst & Leung (1992, ApJS, 82, 167, Equations 2 & 3) */

    else if ( R2 == "THERM" ){

      reaction[reac].k[gridp] = rate_THERM( reac, temperature_gas[gridp],
                                                  temperature_dust[gridp] );
    }


    /* Grain mantle reaction */

    else if ( R2 == "#" ){

      reaction[reac].k[gridp] = rate_GM(reac);
    }





    /* The following 5 rates are described in rate_calculations_radfield.c


    /* Photodesorption */

    else if ( R2 == "PHOTD" ){

      reaction[reac].k[gridp] = rate_PHOTD(reac, temperature_gas[gridp], rad_surface, AV, gridp);
    }


    /* H2 photodissociation */
    /* Taking into account self-shielding and grain extinction */

    else if ( R1 == "H2"  &&  R2 == ""  &&  R3 == "" ){

      H2_photodissociation_nr = reac;

      reaction[reac].k[gridp] = rate_H2_photodissociation( reac, rad_surface, AV, column_H2,
                                                           v_turb, gridp );
    }


    /* HD photodissociation */

    else if ( R1 == "HD"  &&  R2 == ""  &&  R3 == "" ){

      reaction[reac].k[gridp] = rate_H2_photodissociation( reac, rad_surface, AV, column_HD,
                                                           v_turb, gridp );
    }


    /* CO photodissociation */

    else if ( R1 == "CO"  &&  R2 == ""  &&  R3 == ""
              && ( P1 == "C"  ||  P2 == "C"  ||  P3 == "C"  || P4 == "C"  )
              && ( P1 == "O"  ||  P2 == "O"  ||  P3 == "O"  ||  P4 == "O" ) ){

      reaction[reac].k[gridp] = rate_CO_photodissociation( reac, rad_surface, AV,
                                                           column_CO, column_H2, gridp );
    }



    /* C photoionization */

    else if ( R1 == "C"  &&  R2 == ""  &&  R3 == ""
              && ( (P1 == "C+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "C+") ) ){

      C_ionization_nr = reac;

      reaction[reac].k[gridp] = rate_C_photoionization( reac, temperature_gas[gridp], rad_surface,
                                                        AV, column_C, column_H2, gridp );
    }


    /* SI photoionization */

    else if ( R1 == "S"  &&  R2 == ""  &&  R3 == ""
              && ( (P1 == "S+"  &&  P2 == "e-")  ||  (P1 == "e-"  &&  P2 == "S+") ) ){

      reaction[reac].k[gridp] = rate_SI_photoionization(reac, rad_surface, AV, gridp);
    }




    /* The following reactions are again described in rate_calculations.s */


    /* All other reactions */

    else {

      reaction[reac].k[gridp] = rate_canonical(reac, temperature_gas[gridp]);
    }




    /* Now all rates should be calculated */


    /* Check that the rate is physical (0<RATE(I)<1) and produce an error message if not.        */
    /* Impose a lower cut-off on all rate coefficients to prevent the problem becoming too stiff.*/
    /* Rates less than 1E-99 are set to zero.                                                    */
    /* Grain-surface reactions and desorption mechanisms are allowed rates greater than 1.       */

    if (reaction[reac].k[gridp] < 0.0){

      printf("(reaction_rates): ERROR, negative rate for reaction %d \n", reac);
    }

    else if (reaction[reac].k[gridp] > 1.0  &&  R2 != "#"){

      printf("(reaction_rates): WARNING, rate too large for reaction %d \n", reac);
      printf("(reaction_rates): WARNING, rate is set to 1.0 \n");

      reaction[reac].k[gridp] = 1.0;
    }

    else if ( reaction[reac].k[gridp] < 1.0E-99 ){

      reaction[reac].k[gridp] = 0.0;
    }


  } /* end of reac loop over reactions */

}

/*-----------------------------------------------------------------------------------------------*/