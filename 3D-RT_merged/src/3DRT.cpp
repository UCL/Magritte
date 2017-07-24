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

#include "definitions.hpp"
#include "species_tools.cpp"

#include "read_input.cpp"
#include "read_chemdata.cpp"
#include "create_healpixvectors.cpp"
#include "ray_tracing.cpp"
#include "read_linedata.cpp"
#include "calc_C_coeff.cpp"
#include "level_populations.cpp"
#include "column_density_calculator.cpp"
#include "UV_field_calculator.cpp"
#include "write_output.cpp"



/* main code for 3D-RT: 3D Radiative Transfer                                                    */
/*-----------------------------------------------------------------------------------------------*/

int main()
{

  long   n, n1, n2, r, kr;                                                              /* index */

  int    spec;                                                      /* index of chemical species */

  double theta_crit=1.0;           /* critical angle to include a grid point as evaluation point */

  double ray_separation2=0.00;    /* rays closer than the sqrt of this are considered equivalent */

  bool   sobolev=false;               /* Use the Sobolev (large velocity gradient approximation) */

  double time_rt=0.0;                                                     /* time in ray_tracing */
  double time_lp=0.0;                                               /* time in level_populations */



  printf("                             \n");
  printf("3D-RT : 3D Radiative Transfer\n");
  printf("-----------------------------\n");
  printf("                             \n");




  /*   READ INPUT GRID                                                                           */ 
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading grid input\n");



  /* Define grid (using types defined in definitions.h)*/

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */


  /* Initialise (remove garbage from the variables) */

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

  void read_input(string grid_inputfile, GRIDPOINT *gridpoint );

  read_input(grid_inputfile, gridpoint);


  /*_____________________________________________________________________________________________*/





  /*   READ INPUT CHEMISTRY                                                                      */
  /*_____________________________________________________________________________________________*/

  
  /* Read the species and their abundances */

  void read_species(string spec_datafile);

  read_species(spec_datafile);





  /* Setup the (unit) HEALPix vectors */

  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  void create_healpixvectors(double *unit_healpixvector, long *antipod);

  create_healpixvectors(unit_healpixvector, antipod);



  printf("(3D-RT): input read\n");
  printf("(3D-RT): start ray tracing\n\n");


  /* Execute ray_tracing */

  void ray_tracing( double theta_crit, double ray_separation2, double *unit_healpixvector,
                    GRIDPOINT *gridpoint, EVALPOINT *evalpoint);

  time_rt -= omp_get_wtime();

  ray_tracing( theta_crit, ray_separation2, unit_healpixvector, gridpoint, evalpoint);

  time_rt += omp_get_wtime();

  printf("\n(3D-RT): time in ray_tracing: %lf sec\n", time_rt);
  printf("(3D-RT): rays traced\n\n");

  printf("(3D-RT): reading radiative data files\n\n");





  /* --- CHEMISTRY --- */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  /* TEMPORARY CHECK */

  double temperature[NGRID];



  for (n=0; n<NGRID; n++){

    temperature[n] = 10.0; // 10.0*(100.0 + n/NGRID);
    gridpoint[n].density = 10.0;
  }

  /*----------------*/


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /* Read data files */

  line_datafile[0] = LINE_DATAFILE;
  // line_datafile[0] = "data/12c+.dat";
  // line_datafile[0] = "data/12co.dat";
  // line_datafile[0] = "data/16o.dat";


  /* Define and allocate memory for the data */

  int i,j;                                                                      /* level indices */

  int par1, par2, par3;                                         /* index for a collision partner */

  int lspec;                                    /* index of the line species under consideration */


  /* Get the number of levels and cumulatives for each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    nlev[lspec] = get_nlev(line_datafile[lspec]);

    cum_nlev[lspec] = 0;

    cum_nlev2[lspec] = 0;

    printf("(read_linedata): number of energy levels %d\n", nlev[lspec]);
  }


  /* Get the number of radiative transitions and cumulatives for each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    nrad[lspec] = get_nrad(line_datafile[lspec]);

    cum_nrad[lspec] = 0;

    printf("(read_linedata): number of radiative transitions %d\n", nrad[lspec]);
  }


  /* Calculate the cumulatives for nlev and nrad (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_nlev[lspec] = cum_nlev[lspec-1] + nlev[lspec-1];

    cum_nrad[lspec] = cum_nrad[lspec-1] + nrad[lspec-1];

    cum_nlev2[lspec] = cum_nlev2[lspec-1] + nlev[lspec-1]*nlev[lspec-1];
  }



  /* Allocate memory based on nlev and nrad */

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



  /* Get the number of collision partners for each species */

  for (lspec=0; lspec<NLSPEC; lspec++){

    ncolpar[lspec] = get_ncolpar(line_datafile[lspec]);

    cum_ncolpar[lspec] = 0;

    printf("(read_linedata): number of collisional partners %d\n", ncolpar[lspec]);
  }


  /* Calculate the cumulative for ncolpar (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_ncolpar[lspec] = cum_ncolpar[lspec-1] + ncolpar[lspec-1];
  }



  /* Initialize the allocated memory */

  for (lspec=0; lspec<NLSPEC; lspec++){

    for (par1=0; par1<ncolpar[lspec]; par1++){

      ncoltran[LSPECPAR(lspec,par1)] = 0;
      cum_ncoltran[LSPECPAR(lspec,par1)] = 0;

      ncoltemp[LSPECPAR(lspec,par1)] = 0;
      cum_ncoltemp[LSPECPAR(lspec,par1)] = 0;

      cum_ncoltrantemp[LSPECPAR(lspec,par1)] = 0;
    }
  }


  /* For each line producing species */

  for (lspec=0; lspec<NLSPEC; lspec++){


    /* For each collision partner */

    for (par2=0; par2<ncolpar[lspec]; par2++){


      /* Get the number of collisional transitions */

      ncoltran[LSPECPAR(lspec,par2)] = get_ncoltran(line_datafile[lspec], ncoltran, lspec);
/*
      printf( "(read_linedata): number of collisional transitions for partner %d is %d\n",
              par2, ncoltran[LSPECPAR(lspec,par2)] );
*/


      /* Get the number of collision temperatures */

      ncoltemp[LSPECPAR(lspec,par2)] = get_ncoltemp(line_datafile[lspec], ncoltran, par2, lspec);

/*
      printf( "(read_linedata): number of collisional temperatures for partner %d is %d\n",
              par2, ncoltemp[LSPECPAR(lspec,par2)] );
*/
    } /* end of par2 loop over collision partners */

  } /* end of lspec loop over line producing species */


  /* Calculate the cumulatives (needed for indexing, see definitions.h) */

  for (lspec=0; lspec<NLSPEC; lspec++){

    for (par3=1; par3<ncolpar[lspec]; par3++){

      cum_ncoltran[LSPECPAR(lspec,par3)] = cum_ncoltran[LSPECPAR(lspec,par3-1)]
                                             + ncoltran[LSPECPAR(lspec,par3-1)];

      cum_ncoltemp[LSPECPAR(lspec,par3)] = cum_ncoltemp[LSPECPAR(lspec,par3-1)]
                                             + ncoltemp[LSPECPAR(lspec,par3-1)];

      cum_ncoltrantemp[LSPECPAR(lspec,par3)] = cum_ncoltrantemp[LSPECPAR(lspec,par3-1)]
                                                 + ( ncoltran[LSPECPAR(lspec,par3-1)]
                                                     *ncoltemp[LSPECPAR(lspec,par3-1)] );
/*
      printf("(3D-RT): cum_ncoltran[%d] = %d \n", par3, cum_ncoltran[LSPECPAR(lspec,par3)]);
      printf("(3D-RT): cum_ncoltemp[%d] = %d \n", par3, cum_ncoltemp[LSPECPAR(lspec,par3)]);
      printf( "(3D-RT): cum_ncoltrantemp[%d] = %d \n",
              par3, cum_ncoltrantemp[LSPECPAR(lspec,par3)] );
*/
    }
  }


  for (lspec=0; lspec<NLSPEC; lspec++){

    tot_ncoltran[lspec] = cum_ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)]
                          + ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltemp[lspec] = cum_ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)]
                           + ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltrantemp[lspec] = cum_ncoltrantemp[LSPECPAR(lspec,ncolpar[lspec]-1)]
                              + ( ncoltran[LSPECPAR(lspec,ncolpar[lspec]-1)]
         	                  		  *ncoltemp[LSPECPAR(lspec,ncolpar[lspec]-1)] );
/*
    printf("(3D-RT): tot_ncoltran %d\n", tot_ncoltran[lspec]);
    printf("(3D-RT): tot_ncoltemp %d\n", tot_ncoltemp[lspec]);
    printf("(3D-RT): tot_ncoltrantemp %d\n", tot_ncoltrantemp[lspec]);
*/

    cum_tot_ncoltran[lspec] = 0;

    cum_tot_ncoltemp[lspec] = 0;

    cum_tot_ncoltrantemp[lspec] = 0;
  }


  /* Calculate the cumulatives of the cumulatives (also needed for indexing, see definitions.h) */

  for (lspec=1; lspec<NLSPEC; lspec++){

    cum_tot_ncoltran[lspec] = cum_tot_ncoltran[lspec-1] + tot_ncoltran[lspec-1];

    cum_tot_ncoltemp[lspec] = cum_tot_ncoltemp[lspec-1] + tot_ncoltemp[lspec-1];

    cum_tot_ncoltrantemp[lspec] = cum_tot_ncoltrantemp[lspec-1] + tot_ncoltrantemp[lspec-1];
  }



  /* Define and allocate the collision related variables */

  double coltemp[TOT_CUM_TOT_NCOLTEMP];              /* Collision temperatures for each partner */
                                                                  /*[NLSPEC][ncolpar][ncoltemp] */

  double C_data[TOT_CUM_TOT_NCOLTRANTEMP];          /* C_data for each partner, tran. and temp. */
                                                       /* [NLSPEC][ncolpar][ncoltran][ncoltemp] */

  int icol[TOT_CUM_TOT_NCOLTRAN];    /* level index corresp. to col. transition [0..ncoltran-1] */
                                                                 /* [NLSPEC][ncolpar][ncoltran] */

  int jcol[TOT_CUM_TOT_NCOLTRAN];    /* level index corresp. to col. transition [0..ncoltran-1] */





  for(int ind=0; ind<TOT_NCOLPAR; ind++){

    spec_par[ind] = 0;

    ortho_para[ind] = 'i';
  }


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
  }


  for(lspec=0; lspec<NLSPEC; lspec++){

    for (n=0; n<NGRID; n++){

      for (i=0; i<nlev[lspec]; i++){

        for (j=0; j<nlev[lspec]; j++){

	  R[LSPECGRIDLEVLEV(lspec,n,i,j)] = 0.0;
	}
      }
    }
  }

   for(lspec=0; lspec<NLSPEC; lspec++){

    for (kr=0; kr<nrad[lspec]; kr++){

      irad[LSPECRAD(lspec,kr)] = 0;
      jrad[LSPECRAD(lspec,kr)] = 0;
    }
  }


  void read_linedata( string line_datafile, int *irad, int *jrad, double *energy, double *weight,
                      double *frequency, double *A_coeff, double *B_coeff, double *coltemp,
                      double *C_data, int *icol, int *jcol, int lspec );


  for(lspec=0; lspec<NLSPEC; lspec++){

    read_linedata( line_datafile[lspec], irad, jrad, energy, weight, frequency,
                   A_coeff, B_coeff, coltemp, C_data, icol, jcol, lspec );
  }



  // for(int ind=0; ind<ncolpar[0]; ind++){

  //   cout << "spec_par[" << ind << "] = " << spec_par[LSPECPAR(0,ind)] << " (o/p?)" << ortho_para[LSPECPAR(0,ind)] << " \n" ;
  // }



  /* Initializing populations */

  for (lspec=0; lspec<NLSPEC; lspec++){

    for (n=0; n<NGRID; n++){

      for (i=0; i<nlev[lspec]; i++){

        pop[LSPECGRIDLEV(lspec,n,i)] = exp(-HH*CC*energy[LSPECLEV(lspec,i)]/(KB*temperature[n]));

        if(n==5){printf("pop %lE energy %lE \n", pop[LSPECGRIDLEV(lspec,n,i)], energy[LSPECLEV(lspec,i)] );}
      }
    }
  }



  double P_intensity[NGRID*NRAYS];                                   /* Feautrier's mean intensity for a ray */


  for (n1=0; n1<NGRID; n1++){

    for (r=0; r<NRAYS; r++){

      P_intensity[RINDEX(n1,r)] = 0.0;
    }
  }



  printf("\n(3D-RT): start radiative transfer\n\n");


  /* Radiative Transfer: calculate level populations */

  void level_populations( long *antipod, GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                          int *irad, int*jrad, double *frequency, double *A_coeff,
                          double *B_coeff, double *C_coeff, double *P_intensity,
                          double *R, double *pop, double *dpop, double *C_data,
                          double *coltemp, int *icol, int *jcol, double *temperature,
                          double *weight, double *energy, int lspec, bool sobolev );

  time_lp -= omp_get_wtime();

  for (lspec=0; lspec<NLSPEC; lspec++){

    level_populations( antipod, gridpoint, evalpoint, irad, jrad, frequency,
                       A_coeff, B_coeff, C_coeff, P_intensity, R, pop, dpop, C_data,
                       coltemp, icol, jcol, temperature, weight, energy, lspec, sobolev );
  }

  time_lp += omp_get_wtime();


  printf("\n(3D-RT): time in level_populations: %lf sec\n", time_lp);
  printf("(3D-RT): transfer done\n\n");

  printf("(3D-RT): Calculate column densities\n");
  

  printf("(3D-RT): Column densities calculated\n");



  double column_density[NGRID*NSPEC*NRAYS];       /* column density for each spec, ray and gridp */

  double rad_surface[NGRID*NRAYS];

  double AV[NGRID*NRAYS];                       /* Visual extinction (only takes into account H) */


  metallicity = 1.0;

  double UV_field[NGRID*NRAYS];



  /* Initialization */

  for (n=0; n<NGRID; n++){

    for (r=0; r<NRAYS; r++){

      UV_field[RINDEX(n,r)]    = 0.0;
      rad_surface[RINDEX(n,r)] = 0.0;

      for (spec=0; spec<NSPEC; spec++){

        column_density[GRIDSPECRAY(n,spec,r)] = 0.0;
      }

    }
  }


  void column_density_calculator( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                  double *column_density, double *AV );

  column_density_calculator( gridpoint, evalpoint, column_density, AV );


  double G_external[3];                                              /* external radiation field */

  G_external[0] = 0.0;
  G_external[1] = 0.0;
  G_external[2] = 0.0;


  void UV_field_calculator(double *G_external, double *UV_field, double *rad_surface);

  UV_field_calculator(G_external, UV_field, rad_surface);





  /*   WRITE OUTPUT                                                                              */
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): writing output \n");


  /* Write the output file  */

  void write_output( double *unit_healpixvector, long *antipod, GRIDPOINT *gridpoint,
                     EVALPOINT *evalpoint, double *pop, double *weight, double *energy );

  write_output( unit_healpixvector, antipod, gridpoint, evalpoint, pop, weight, energy );


  printf("(3D-RT): output written \n\n");


  /*_____________________________________________________________________________________________*/





  printf("(3D-RT): done\n");


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
