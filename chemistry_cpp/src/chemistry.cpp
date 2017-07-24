/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*                                                                                               */
/* CHEMISTRY                                                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>

#include <string>
using namespace std;

#include "definitions.hpp"
#include "read_chemdata.cpp"
#include "reaction_rates.cpp"
#include "abundance.cpp"




  /*--- TEMPORARY CHEMISTRY ---*/

int main(){

  ngrid = 10;

  int n, spec, reac, ray;                                                               /* index */


  /* Specify the file containing the species */

  string specdatafile = "data/species_reduced.d";    /* path to data file containing the species */


  /* Specify the file containing the reactions */

  string reacdatafile = "data/rates_reduced.d";    /* path to data file containing the reactions */


  /* Get the number of species from the species data file */

  nspec = get_nspec(specdatafile);
  printf("(read_chemdata): number of species   %*d\n", MAX_WIDTH, nspec);


  /* Get the number of reactions from the reactions data file */

  nreac = get_nreac(reacdatafile);
  printf("(read_chemdata): number of reactions %*d\n", MAX_WIDTH, nreac);


  /* Declare the species and reactions */

  // SPECIES species[nspec];     /* species has a symbol (.sym), mass (.mass), and abundance (.abn) */

  species = (SPECIES*) malloc( nspec*sizeof(SPECIES) );


  // REACTIONS reaction[nreac];         /* reaction has reactants (.R1, .R2, .R3), reaction products \
                                (.P1, .P2, .P3, .P4), alpha (.alpha), beta (.beta), gamma (.gamma)\
                                minimal temperature (.RT_min) and maximal temperature (RT_max) */
  // reaction = (REACTIONS*) malloc( nreac*sizeof(REACTIONS) );


  void read_species(string specdatafile);

  read_species(specdatafile);


  void read_reactions(string reacdatafile, REACTIONS *reaction);

  read_reactions(reacdatafile, reaction);


  for(spec=0; spec<nspec; spec++){

    printf( "%s\t%.2lE\t%.1lf\n",
            species[spec].sym.c_str(), species[spec].abn[0], species[spec].mass );
  }

  for(reac=0; reac<nreac; reac++){

    printf( "%-7s+%-7s+%-7s  ->  %-7s+%-7s+%-7s+%-7s"
            "with alpha = %-10.2lE, beta = %-10.2lE, gamma = %-10.2lE \t"
            "RT_min = %-10.2lE, RT_max = %-10.2lE, duplicates = %d \n",
            reaction[reac].R1.c_str(), reaction[reac].R2.c_str(), reaction[reac].R3.c_str(),
            reaction[reac].P1.c_str(), reaction[reac].P2.c_str(), reaction[reac].P3.c_str(),
            reaction[reac].P4.c_str(),
            reaction[reac].alpha, reaction[reac].beta, reaction[reac].gamma,
            reaction[reac].RT_min, reaction[reac].RT_max,
            reaction[reac].dup );
  }
  
  double temperature_gas = 1.0;
  double temperature_dust = 0.1;
  double gas2dust = 100.0;
  double metallicity = 1.0;

  double v_turb=1.0E-4;                                            /* turbulent speed of the gas */

  double *rad_surface;
  rad_surface = (double*) malloc( NRAYS*sizeof(double) );

  double *AV;
  AV = (double*) malloc( NRAYS*sizeof(double) );

  double *column_H2;
  column_H2 = (double*) malloc( NRAYS*sizeof(double) );

  double *column_HD;
  column_HD = (double*) malloc( NRAYS*sizeof(double) );

  double *column_CI;
  column_CI = (double*) malloc( NRAYS*sizeof(double) );

  double *column_CO;
  column_CO = (double*) malloc( NRAYS*sizeof(double) );

  double *UV_field;
  UV_field = (double*) malloc( ngrid*sizeof(double) );




  /* TEMPORARY: write proper version when coupling to transfer code */
  /* -------------------------------------------------------------- */

  for (ray=0; ray<NRAYS; ray++){

    rad_surface[ray] = 1.0;
    AV[ray]          = 1.0;
    column_H2[ray]   = 1.0;
    column_HD[ray]   = 1.0;
    column_CI[ray]   = 1.0;
    column_CO[ray]   = 1.0;
  }

  for (n=0; n<ngrid; n++){

    UV_field[n] = 1.0;
  }

  /* -------------------------------------------------------------- */




  /* Calculate the dust temperature */

  void dust_temperature_calculation( double *UV_field, double *rad_surface,
                                     double *temperature_dust );



  /* Calculate the reaction k coefficients from the reaction data */

  void reaction_rates( REACTIONS *reaction, double temperature_gas, double temperature_dust,
                       double metallicity, double gas2dust, double *rad_surface, double *AV,
                       double *column_H2, double *column_HD, double *column_CI, double *column_CO,
                       double v_turb );

  reaction_rates( reaction, temperature_gas, temperature_dust, metallicity, gas2dust,
                  rad_surface, AV, column_H2, column_HD, column_CI, column_CO, v_turb );




  for (n=0; n<ngrid; n++){

    for (spec=0; spec<nspec; spec++){

      if ( (species[spec].sym == "H2")  ||  (species[spec].sym == "H")
           ||  (species[spec].sym == "He") ||  (species[spec].sym == "e-") ){

        species[spec].abn[n] = species[spec].abn[n];
      }

      else {

        species[spec].abn[n] = species[spec].abn[n] * metallicity;
      }
    }
  }

  void abundance();

  abundance();


  return(0);
}
