/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* read_chemdata: Read the chemical data files                                                   */
/*                                                                                               */
/* (based on read_species and read_rates in 3D-PDR)                                              */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <string>
#include <iostream>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "read_chemdata.hpp"
#include "species_tools.hpp"



/* read_species: read the species from the data file                                             */
/*-----------------------------------------------------------------------------------------------*/

int read_species(std::string spec_datafile)
{


  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */

  char sym_buff[15];

  double abn_buff;
  double mass_buff;



  /* The first species is a dummy for when a species is not found */

  species[0].sym = "dummy0";

  for (long gridp=0; gridp<NGRID; gridp++){

    species[0].abn[gridp] = 0.0;
  }


  /* The last species is a dummy with abundance 1.0 everywhere */

  species[NSPEC-1].sym = "dummy1";

  for (long gridp=0; gridp<NGRID; gridp++){

    species[NSPEC-1].abn[gridp] = 1.0;
  }



  /* Open species data file */

  FILE *specdata = fopen(spec_datafile.c_str(), "r");


  for (int l=1; l<NSPEC-1; l++){

    int n;

    fgets(buffer, BUFFER_SIZE, specdata);
    sscanf( buffer, "%d,%[^,],%lE,%lf %*[^\n] \n", &n, sym_buff, &abn_buff, &mass_buff );

    species[l].sym  = sym_buff;
    species[l].mass = mass_buff;


    for (long gridp=0; gridp<NGRID; gridp++){

      species[l].abn[gridp] = abn_buff;
    }
  }


  /* Overwrite electron abindance */

  printf("\n\nNote: The electron abundance will be overwritten to make the gas neutral \n\n");

  int electron_nr = get_species_nr("e-");

  for (long gridp=0; gridp<NGRID; gridp++){

    species[electron_nr].abn[gridp] = 0.0;
    species[electron_nr].abn[gridp] = get_electron_abundance(gridp);
  }

  fclose(specdata);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* read_reactions: read the reactoins from the (CSV) data file                                   */
/*-----------------------------------------------------------------------------------------------*/

int read_reactions(std::string reac_datafile)
{


  char *buffer;                                                     /* buffer for a line of data */
  buffer = (char*) malloc( BUFFER_SIZE*sizeof(char) );

  char *buffer_cpy;                                                 /* buffer for a line of data */
  buffer_cpy = (char*) malloc( BUFFER_SIZE*sizeof(char) );


  char *alpha_buff;
  alpha_buff = (char*) malloc( 15*sizeof(char) );

  char *beta_buff;
  beta_buff = (char*) malloc( 15*sizeof(char) );

  char *gamma_buff;
  gamma_buff = (char*) malloc( 15*sizeof(char) );

  char *RT_min_buff;
  RT_min_buff = (char*) malloc( 15*sizeof(char) );

  char *RT_max_buff;
  RT_max_buff = (char*) malloc( 15*sizeof(char) );


  /* Open reactions data file */

  FILE *reacdata = fopen(reac_datafile.c_str(), "r");


  for (int l=0; l<NREAC; l++){

    fgets(buffer, BUFFER_SIZE, reacdata);

    buffer_cpy = strdup(buffer);


    /* Ignore first column */

    strsep(&buffer_cpy, ",");


    /* Read first reactant */

    reaction[l].R1 = strsep(&buffer_cpy, ",");


    /* Read second reactant */

    reaction[l].R2 = strsep(&buffer_cpy, ",");


    /* Read third reactant */

    reaction[l].R3 = strsep(&buffer_cpy, ",");


    /* Read first reaction product */

    reaction[l].P1 = strsep(&buffer_cpy, ",");


    /* Read second reaction product */

    reaction[l].P2 = strsep(&buffer_cpy, ",");


    /* Read third reaction product */

    reaction[l].P3 = strsep(&buffer_cpy, ",");


    /* Read fourth reaction product */

    reaction[l].P4 = strsep(&buffer_cpy, ",");


    /* Read alpha */

    alpha_buff = strsep(&buffer_cpy, ",");
    sscanf(alpha_buff, "%lE", &reaction[l].alpha);


    /* Read beta */

    beta_buff = strsep(&buffer_cpy, ",");
    sscanf(beta_buff, "%lf", &reaction[l].beta);


    /* Read gamma */

    gamma_buff = strsep(&buffer_cpy, ",");
    sscanf(gamma_buff, "%lf", &reaction[l].gamma);


    /* Ignore next column */

    strsep(&buffer_cpy, ",");


    RT_min_buff = strsep(&buffer_cpy, ",");
    sscanf(RT_min_buff, "%lf", &reaction[l].RT_min);


    /* Read RT_max */

    RT_max_buff = strsep(&buffer_cpy, ",");
    sscanf(RT_max_buff, "%lf", &reaction[l].RT_max);


    /* Check for duplicates */
    /* reaction[reac].dup are the number of instances of the same reaction as reac before reac */
    /* This is different from 3D-PDR */

    reaction[l].dup = 0;

    for (int reac=0; reac<l; reac++){

      if ( reaction[l].R1 == reaction[reac].R1
           &&  reaction[l].R2 == reaction[reac].R2
           &&  reaction[l].R3 == reaction[reac].R3
           &&  reaction[l].P1 == reaction[reac].P1
           &&  reaction[l].P2 == reaction[reac].P2
           &&  reaction[l].P3 == reaction[reac].P3
           &&  reaction[l].P4 == reaction[reac].P4 ){

        reaction[l].dup = reaction[l].dup + 1;

      }
    }

  }

  fclose(reacdata);


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
