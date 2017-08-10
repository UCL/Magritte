/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* 3D-RT: main                                                                                   */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
/*#include <mpi.h>*/

#include <string>
#include <iostream>
using namespace std;

#include "declarations.hpp"
#include "definitions.hpp"

#include "species_tools.hpp"
#include "data_tools.hpp"
#include "setup_data_structures.hpp"

#include "read_input.hpp"
#include "read_chemdata.hpp"
#include "read_linedata.hpp"

#include "create_healpixvectors.hpp"
#include "ray_tracing.hpp"

#include "rad_surface_calculator.hpp"
#include "column_density_calculator.hpp"
#include "AV_calculator.hpp"
#include "UV_field_calculator.hpp"
#include "dust_temperature_calculation.hpp"
#include "abundances.hpp"
#include "level_populations.hpp"

#include "write_output.hpp"



/* main code for 3D-RT: 3D Radiative Transfer                                                    */
/*-----------------------------------------------------------------------------------------------*/

int main()
{

  long   n     = 0;                                                          /* grid point index */
  long   n1    = 0;                                                          /* grid point index */
  long   n2    = 0;                                                          /* grid point index */
  long   r     = 0;                                                                 /* ray index */

  int    i     = 0;                                                    /* population level index */
  int    j     = 0;                                                    /* population level index */
  int    kr    = 0;                                             /* index of radiative transition */
  int    spec  = 0;                                                 /* index of chemical species */
  int    lspec = 0;                                                     /* index of line species */
  int    par   = 0;                                                /* index of collision partner */

  double time_ray_tracing = 0.0;                                    /* total time in ray_tracing */
  double time_abundances = 0.0;                                      /* total time in abundances */
  double time_level_pop = 0.0;                                /* total time in level_populations */


  /* Temporary */

  metallicity = 1.0;

  gas2dust = 100.0;

  double v_turb = 0.0;





  printf("                              \n");
  printf("3D-RT : 3D Radiative Transfer \n");
  printf("----------------------------- \n");
  printf("                              \n");





  /*   READ GRID INPUT                                                                           */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading grid input \n");


  /* Define grid (using types defined in definitions.h)*/

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */



  /* Initialize */

  for (n1=0; n1<NGRID; n1++){

    for (n2=0; n2<NGRID; n2++){

      evalpoint[GINDEX(n1,n2)].dZ  = 0.0;
      evalpoint[GINDEX(n1,n2)].Z   = 0.0;
      evalpoint[GINDEX(n1,n2)].vol = 0.0;

      evalpoint[GINDEX(n1,n2)].ray = 0;
      evalpoint[GINDEX(n1,n2)].nr  = 0;

      evalpoint[GINDEX(n1,n2)].eqp = 0;

      evalpoint[GINDEX(n1,n2)].onray = false;

      key[GINDEX(n1,n2)] = 0;
    }

    for (r=0; r<NRAYS; r++){

      raytot[RINDEX(n1,r)]      = 0;
      cum_raytot[RINDEX(n1,r)]  = 0;
    }

  }



  /* Read input file */

  read_input(grid_inputfile, gridpoint);


  printf("(3D-RT): grid input read \n\n");


  /*_____________________________________________________________________________________________*/





  /*   READ INPUT CHEMISTRY                                                                      */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading chemistry input \n");


  /* Read the species (and their initial abundances) */

  read_species(spec_datafile);


  e_nr    = get_species_nr("e-");                       /* species nr corresponding to electrons */
  H2_nr   = get_species_nr("H2");                              /* species nr corresponding to H2 */
  HD_nr   = get_species_nr("HD");                              /* species nr corresponding to HD */
  C_nr    = get_species_nr("C");                                /* species nr corresponding to C */
  H_nr    = get_species_nr("H");                                /* species nr corresponding to H */
  H2x_nr  = get_species_nr("H2+");                            /* species nr corresponding to H2+ */
  HCOx_nr = get_species_nr("HCO+");                          /* species nr corresponding to HCO+ */
  H3x_nr  = get_species_nr("H3+");                            /* species nr corresponding to H3+ */
  H3Ox_nr = get_species_nr("H3O+");                          /* species nr corresponding to H3O+ */
  Hex_nr  = get_species_nr("He+");                            /* species nr corresponding to He+ */
  CO_nr   = get_species_nr("CO");                              /* species nr corresponding to CO */


  /* Read the reactions */

  read_reactions(reac_datafile);


  printf("(3D-RT): chemistry input read \n\n");


  /*_____________________________________________________________________________________________*/





  /*   CREATE HEALPIX VEXTORS AND FIND ANTIPODAL PAIRS                                           */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): creating HEALPix vectors \n");


  /* Create the (unit) HEALPix vectors and find antipodal pairs */

  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  create_healpixvectors(unit_healpixvector, antipod);


  printf("(3D-RT): HEALPix vectors creatied \n\n");


  /*_____________________________________________________________________________________________*/





  /*   RAY TRACING                                                                               */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): ray tracing \n");


  /* Execute ray_tracing */

  time_ray_tracing -= omp_get_wtime();

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);

  time_ray_tracing += omp_get_wtime();


  printf("\n(3D-RT): time in ray_tracing: %lf sec \n", time_ray_tracing);


  printf("(3D-RT): rays traced \n\n");


  /*_____________________________________________________________________________________________*/





  /*   SETUP DATA STRUCTURES FOR LINE DATA                                                       */
  /*_____________________________________________________________________________________________*/


  setup_data_structures(line_datafile);



  /* Define line related variables */

  int irad[TOT_NRAD];           /* level index corresponding to radiative transition [0..nrad-1] */

  int jrad[TOT_NRAD];           /* level index corresponding to radiative transition [0..nrad-1] */

  double energy[TOT_NLEV];                                                /* energy of the level */

  double weight[TOT_NLEV];                                    /* statistical weight of the level */

  double frequency[TOT_NLEV2];             /* photon frequency corresponing to i -> j transition */

  double A_coeff[TOT_NLEV2];                                        /* Einstein A_ij coefficient */

  double B_coeff[TOT_NLEV2];                                        /* Einstein B_ij coefficient */

  double C_coeff[TOT_NLEV2];                                        /* Einstein C_ij coefficient */

  double R[NGRID*TOT_NLEV2];                                           /* transition matrix R_ij */

  double pop[NGRID*TOT_NLEV];                                            /* level population n_i */

  double dpop[NGRID*TOT_NLEV];        /* change in level population n_i w.r.t previous iteration */



  /* Define the collision related variables */

  double coltemp[TOT_CUM_TOT_NCOLTEMP];               /* Collision temperatures for each partner */
                                                                   /*[NLSPEC][ncolpar][ncoltemp] */

  double C_data[TOT_CUM_TOT_NCOLTRANTEMP];           /* C_data for each partner, tran. and temp. */
                                                        /* [NLSPEC][ncolpar][ncoltran][ncoltemp] */

  int icol[TOT_CUM_TOT_NCOLTRAN];     /* level index corresp. to col. transition [0..ncoltran-1] */
                                                                  /* [NLSPEC][ncolpar][ncoltran] */

  int jcol[TOT_CUM_TOT_NCOLTRAN];     /* level index corresp. to col. transition [0..ncoltran-1] */
                                                                  /* [NLSPEC][ncolpar][ncoltran] */


  /* Initializing data */

  for(lspec=0; lspec<NLSPEC; lspec++){

    for (i=0; i<nlev[lspec]; i++){

      weight[LSPECLEV(lspec,i)] = 0.0;
      energy[LSPECLEV(lspec,i)] = 0.0;

      for (j=0; j<nlev[lspec]; j++){

        A_coeff[LSPECLEVLEV(lspec,i,j)] = 0.0;
        B_coeff[LSPECLEVLEV(lspec,i,j)] = 0.0;
        C_coeff[LSPECLEVLEV(lspec,i,j)] = 0.0;

        frequency[LSPECLEVLEV(lspec,i,j)] = 0.0;

        for (n=0; n<NGRID; n++){

          R[LSPECGRIDLEVLEV(lspec,n,i,j)] = 0.0;
        }
      }
    }


    for (kr=0; kr<nrad[lspec]; kr++){

      irad[LSPECRAD(lspec,kr)] = 0;
      jrad[LSPECRAD(lspec,kr)] = 0;
    }
  }


  /* Initialize */

  for(par=0; par<TOT_NCOLPAR; par++){

    spec_par[par] = 0;

    ortho_para[par] = 'i';
  }


  /*_____________________________________________________________________________________________*/





  /*   READ LINE DATA FOR EACH LINE PRODUCING SPECIES                                            */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading line data \n");


  /* For all line producing species */

  for(lspec=0; lspec<NLSPEC; lspec++){

    read_linedata( line_datafile[lspec], irad, jrad, energy, weight, frequency,
                   A_coeff, B_coeff, coltemp, C_data, icol, jcol, lspec );
  }


  printf("(3D-RT): line data read \n");


  /*_____________________________________________________________________________________________*/





  /*   CALCULATE THE EXTERNAL RADIATION FIELD                                                    */
  /*_____________________________________________________________________________________________*/


  double G_external[3];                                       /* external radiation field vector */

  double rad_surface[NGRID*NRAYS];


  /* Initialize */

  G_external[0] = 0.0;
  G_external[1] = 0.0;
  G_external[2] = 0.0;


  for (n=0; n<NGRID; n++){

    for (r=0; r<NRAYS; r++){

      rad_surface[RINDEX(n,r)] = 0.0;
    }
  }


  /* Calculate the radiation surface */

  rad_surface_calculator(G_external, unit_healpixvector, rad_surface);


  /*_____________________________________________________________________________________________*/





  /*   CALCULATE THERMAL BALANCE (ITERATIVELY)                                                   */
  /*_____________________________________________________________________________________________*/


  bool no_thermal_balance = true;

  double temperature_gas[NGRID];                    /* temperature of the gas at each grid point */
  double temperature_dust[NGRID];                  /* temperature of the dust at each grid point */

  double column_H2[NGRID*NRAYS];                /* H2 column density for each ray and grid point */
  double column_HD[NGRID*NRAYS];                /* HD column density for each ray and grid point */
  double column_C[NGRID*NRAYS];                  /* C column density for each ray and grid point */
  double column_CO[NGRID*NRAYS];                /* CO column density for each ray and grid point */

  double AV[NGRID*NRAYS];                       /* Visual extinction (only takes into account H) */

  double UV_field[NGRID];


  for (n=0; n<NGRID; n++){

    temperature_gas[n] = 10.0;
  }



  /* Initialization */

  for (n=0; n<NGRID; n++){

    UV_field[n] = 0.0;

    for (r=0; r<NRAYS; r++){

      column_H2[RINDEX(n,r)] = 0.0;
      column_HD[RINDEX(n,r)] = 0.0;
      column_C[RINDEX(n,r)]  = 0.0;
      column_CO[RINDEX(n,r)] = 0.0;
    }
  }



  /* Thermal balance iterations */

  while (no_thermal_balance){


    /* Calculate column densities */

    column_density_calculator(gridpoint, evalpoint, column_H2, H2_nr);
    column_density_calculator(gridpoint, evalpoint, column_HD, HD_nr);
    column_density_calculator(gridpoint, evalpoint, column_C, C_nr);
    column_density_calculator(gridpoint, evalpoint, column_CO, CO_nr);


    /* Calculate the visual extinction */

    AV_calculator(column_H2, AV);


    /* Calculcate the UV field */

    UV_field_calculator(AV, rad_surface, UV_field);


    /* Calculate the dust temperature */

    dust_temperature_calculation(UV_field, rad_surface, temperature_dust);





    /*   CALCULATE CHEMICAL ABUNDANCES (ITERATIVELY)                                             */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    printf("(3D-RT): calculating chemical abundances \n\n");


    /* Calculate the chemical abundances by solving the rate equations */

    time_abundances -= omp_get_wtime();

    abundances( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
                column_H2, column_HD, column_C, column_CO, v_turb );

    time_abundances += omp_get_wtime();


    printf("\n(3D-RT): time in abundances: %lf sec\n", time_abundances);


    printf("(3D-RT): chemical abundances calculated \n\n");


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/





    /*   CALCULATE LEVEL POPULATIONS (ITERATIVELY)                                               */
    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    printf("(3D-RT): calculating level populations \n\n");


    /* Initializing populations */

    for (lspec=0; lspec<NLSPEC; lspec++){

      for (n=0; n<NGRID; n++){

        for (i=0; i<nlev[lspec]; i++){

          pop[LSPECGRIDLEV(lspec,n,i)] = exp( -HH*CC*energy[LSPECLEV(lspec,i)]
                                               / (KB*temperature_gas[n]) );
        }
      }
    }


    /* Declare and initialize P_intensity for each ray through a grid point */

    double P_intensity[NGRID*NRAYS];                     /* Feautrier's mean intensity for a ray */


    for (n1=0; n1<NGRID; n1++){

      for (r=0; r<NRAYS; r++){

        P_intensity[RINDEX(n1,r)] = 0.0;
      }
    }


    /* Calculate level populations for each line producing species */

    time_level_pop -= omp_get_wtime();


    /* For each line producing species */

    for (lspec=0; lspec<NLSPEC; lspec++){

      level_populations( antipod, gridpoint, evalpoint, irad, jrad, frequency,
                         A_coeff, B_coeff, C_coeff, P_intensity, R, pop, dpop, C_data,
                         coltemp, icol, jcol, temperature_gas, weight, energy, lspec );
    }


    time_level_pop += omp_get_wtime();


    printf("\n(3D-RT): time in level_populations: %lf sec\n", time_level_pop);


    printf("(3D-RT): level populations calculated \n\n");


    /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


    /* Only one iteration for the moment */

    no_thermal_balance = false;


  } /* end of thermal balance iterations */


  /*_____________________________________________________________________________________________*/





  /*   WRITE OUTPUT                                                                              */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): writing output \n");


  /* Write the output file  */

  write_output( unit_healpixvector, antipod, gridpoint, evalpoint, pop, weight, energy );


  printf("(3D-RT): output written \n\n");


  /*_____________________________________________________________________________________________*/





  printf("(3D-RT): done \n\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
