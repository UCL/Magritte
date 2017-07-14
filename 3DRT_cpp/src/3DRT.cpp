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

void main()
{

  long   n, n1, n2, r, kr;                                                              /* index */

  int    spec;                                                      /* index of chemical species */

  double theta_crit=1.0;           /* critical angle to include a grid point as evaluation point */

  double ray_separation2=0.00;    /* rays closer than the sqrt of this are considered equivalent */

  bool sobolev=false;                 /* Use the Sobolev (large velocity gradient approximation) */


  double time_rt=0.0;                                                     /* time in ray_tracing */
  double time_lp=0.0;                                               /* time in level_populations */

  nsides = 4;                                         /* Defined in HEALPix, NRAYS = 12*nsides^2 */

  double *unit_healpixvector;                    /* array of HEALPix vectors for each ipix pixel */
  unit_healpixvector = (double*) malloc( 3*NRAYS*sizeof(double) );


  long   *antipod;                                           /* gives antipodal ray for each ray */
  antipod = (long*) malloc( NRAYS*sizeof(long) );


  printf("                             \n");
  printf("3D-RT : 3D Radiative Transfer\n");
  printf("-----------------------------\n");
  printf("                             \n");





  /* --- FROM GRIDPOINTS TO EVALUATION POINTS --- */
  /*----------------------------------------------*/





  /*   READ INPUT GRID                                                                           */ 
  /*_____________________________________________________________________________________________*/


  printf("(3D-RT): reading grid input\n");


  /* Specify the input file */

  string inputfile = "input/grid_1D_regular.txt";


  /* Count number of grid points in input file input/ingrid.txt */

  long get_ngrid(string inputfile);                                   /* defined in read_input.c */

  ngrid = get_ngrid(inputfile);                       /* number of grid points in the input file */



  /* Define and allocate memory for grid (using types defined in definitions.h)*/

  GRIDPOINT *gridpoint;                                                           /* grid points */
  gridpoint = (GRIDPOINT*) malloc( ngrid*sizeof(GRIDPOINT) );

  EVALPOINT *evalpoint;                                 /* evaluation points for each grid point */
  evalpoint = (EVALPOINT*) malloc( ngrid*ngrid*sizeof(EVALPOINT) );


  /* Allocate memory for the variables needed to efficiently store the evalpoints */

  cum_raytot = (long*) malloc( ngrid*NRAYS*sizeof(long) );

  key = (long*) malloc( ngrid*ngrid*sizeof(long) );

  raytot = (long*) malloc( ngrid*NRAYS*sizeof(long) );


  /* Initialise (remove garbage out of the variables) */

  for (n1=0; n1<ngrid; n1++){

    for (n2=0; n2<ngrid; n2++){

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

  void read_input(string inputfile, long ngrid, GRIDPOINT *gridpoint );

  read_input(inputfile, ngrid, gridpoint);


  /*_____________________________________________________________________________________________*/




  /*   READ INPUT CHEMISTRY                                                                      */
  /*_____________________________________________________________________________________________*/






  /* Read chemical data */



  /* --- CHEMISTRY --- */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  

  /* Specify the file containing the species */

  string specdatafile = "data/species_reduced.d";    /* path to data file containing the species */


  /* Get the number of species from the species data file */

  nspec = get_nspec(specdatafile);
  printf("(read_chemdata): number of species   %*d\n", MAX_WIDTH, nspec);


  species = (SPECIES*) malloc( nspec*sizeof(SPECIES) );




  int lspec;                                    /* index of the line species under consideration */

  
  /* Read the species and their abundances */

  void read_species(string specdatafile);

  read_species(specdatafile);



  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  /* Setup the (unit) HEALPix vectors */

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

  double *temperature;
  temperature = (double*) malloc( ngrid*sizeof(double) );


  for (n=0; n<ngrid; n++){

    temperature[n] = 10.0; // 10.0*(100.0 + n/ngrid);
    gridpoint[n].density = 10.0;
  }

  /*----------------*/


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /* Read data files */

  nline_species = 1;

  string datafile[nline_species];

  datafile[0] = "data/12c.dat";
  // datafile[0] = "data/12c+.dat";
  // datafile[0] = "data/12co.dat";
  // datafile[0] = "data/16o.dat";


  /* Define and allocate memory for the data */

  int i,j;                                                                      /* level indices */

  int par1, par2, par3;                                         /* index for a collision partner */


  /* Allocate and get the number of energy levels for each line producing species */

  /* int *nlev;                                                       /* number of energy levels */
  nlev = (int*) malloc( nline_species*sizeof(int) );

  /* int *cum_nlev;                                        /* cumulative number of energy levels */
  cum_nlev = (int*) malloc( nline_species*sizeof(int) );

  /* int *cum_nlev2;                          /* cumulative of square of number of energy levels */
  cum_nlev2 = (int*) malloc( nline_species*sizeof(int) );


  for (lspec=0; lspec<nline_species; lspec++){

    nlev[lspec] = get_nlev(datafile[lspec]);

    cum_nlev[lspec] = 0;

    cum_nlev2[lspec] = 0;

    printf("(read_linedata): number of energy levels %d\n", nlev[lspec]);
  }


  /* Allocate and get the number of radiative transitions for each line producing species */

  nrad = (int*) malloc( nline_species*sizeof(int) );

  cum_nrad = (int*) malloc( nline_species*sizeof(int) );

  for (lspec=0; lspec<nline_species; lspec++){

    nrad[lspec] = get_nrad(datafile[lspec]);

    cum_nrad[lspec] = 0;

    printf("(read_linedata): number of radiative transitions %d\n", nrad[lspec]);
  }


  /* Calculate the cumulatives for nlev and nrad (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<nline_species; lspec++){

    cum_nlev[lspec] = cum_nlev[lspec-1] + nlev[lspec-1];

    cum_nrad[lspec] = cum_nrad[lspec-1] + nrad[lspec-1];

    cum_nlev2[lspec] = cum_nlev2[lspec-1] + nlev[lspec-1]*nlev[lspec-1];
  }

  tot_nlev = cum_nlev[nline_species-1] + nlev[nline_species-1];           /* tot. nr. of levels */

  tot_nrad = cum_nrad[nline_species-1] + nrad[nline_species-1];      /* tot. nr. of transitions */

  tot_nlev2 = cum_nlev2[nline_species-1] + nlev[nline_species-1]*nlev[nline_species-1];
                                                              /* tot of squares of nr of levels */


  /* Allocate memory based on nlev and nrad */

  int *irad;                    /* level index corresponding to radiative transition [0..nrad-1] */
  irad = (int*) malloc( tot_nrad*sizeof(int) );

  int *jrad;                    /* level index corresponding to radiative transition [0..nrad-1] */
  jrad = (int*) malloc( tot_nrad*sizeof(int) );

  double *energy;                                                         /* energy of the level */
  energy = (double*) malloc( tot_nlev*sizeof(double) );

  double *weight;                                             /* statistical weight of the level */
  weight = (double*) malloc( tot_nlev*sizeof(double) );

  double *frequency;                       /* photon frequency corresponing to i -> j transition */
  frequency = (double*) malloc( tot_nlev2*sizeof(double) );

  double *A_coeff;                                                  /* Einstein A_ij coefficient */
  A_coeff = (double*) malloc( tot_nlev2*sizeof(double) );

  double *B_coeff;                                                  /* Einstein B_ij coefficient */
  B_coeff = (double*) malloc( tot_nlev2*sizeof(double) );

  double *C_coeff;                                                  /* Einstein C_ij coefficient */
  C_coeff = (double*) malloc( tot_nlev2*sizeof(double) );

  double *R;                                                           /* transition matrix R_ij */
  R = (double*) malloc( ngrid*tot_nlev2*sizeof(double) );

  double *pop;                                                           /* level population n_i */
  pop = (double*) malloc( ngrid*tot_nlev*sizeof(double) );

  double *dpop;                       /* change in level population n_i w.r.t previous iteration */
  dpop = (double*) malloc( ngrid*tot_nlev*sizeof(double) );



  /* Get the number of collision partners for each species */

  ncolpar = (int*) malloc( nline_species*sizeof(int) );

  cum_ncolpar = (int*) malloc( nline_species*sizeof(int) );

  for (lspec=0; lspec<nline_species; lspec++){

    ncolpar[lspec] = get_ncolpar(datafile[lspec]);

    cum_ncolpar[lspec] = 0;

    printf("(read_linedata): number of collisional partners %d\n", ncolpar[lspec]);
  }


  /* Calculate the cumulative for ncolpar (needed for indexing, see definitions.h) */

  for (lspec=1; lspec<nline_species; lspec++){

    cum_ncolpar[lspec] = cum_ncolpar[lspec-1] + ncolpar[lspec-1];
  }

  tot_ncolpar = cum_ncolpar[nline_species-1] + ncolpar[nline_species-1];


  /* Allocate memory based on ncolpar */

  ncoltran = (int*) malloc( tot_ncolpar*sizeof(int) );

  cum_ncoltran = (int*) malloc( tot_ncolpar*sizeof(int) );

  tot_ncoltran = (int*) malloc( nline_species*sizeof(int) );

  cum_tot_ncoltran = (int*) malloc( nline_species*sizeof(int) );

  ncoltemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  cum_ncoltemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  tot_ncoltemp = (int*) malloc( nline_species*sizeof(int) );

  cum_tot_ncoltemp = (int*) malloc( nline_species*sizeof(int) );

  cum_ncoltrantemp = (int*) malloc( tot_ncolpar*sizeof(int) );

  tot_ncoltrantemp = (int*) malloc( nline_species*sizeof(int) );

  cum_tot_ncoltrantemp = (int*) malloc( nline_species*sizeof(int) );


  /* Initialize the allocated memory */

  for (lspec=0; lspec<nline_species; lspec++){

    for (par1=0; par1<ncolpar[lspec]; par1++){

      ncoltran[SPECPAR(lspec,par1)] = 0;
      cum_ncoltran[SPECPAR(lspec,par1)] = 0;

      ncoltemp[SPECPAR(lspec,par1)] = 0;
      cum_ncoltemp[SPECPAR(lspec,par1)] = 0;

      cum_ncoltrantemp[SPECPAR(lspec,par1)] = 0;
    }
  }


  /* For each line producing species */

  for (lspec=0; lspec<nline_species; lspec++){


    /* For each collision partner */

    for (par2=0; par2<ncolpar[lspec]; par2++){


      /* Get the number of collisional transitions */

      ncoltran[SPECPAR(lspec,par2)] = get_ncoltran(datafile[lspec], ncoltran, lspec);
/*
      printf( "(read_linedata): number of collisional transitions for partner %d is %d\n",
              par2, ncoltran[SPECPAR(lspec,par2)] );
*/


      /* Get the number of collision temperatures */

      ncoltemp[SPECPAR(lspec,par2)] = get_ncoltemp(datafile[lspec], ncoltran, par2, lspec);

/*
      printf( "(read_linedata): number of collisional temperatures for partner %d is %d\n",
              par2, ncoltemp[SPECPAR(lspec,par2)] );
*/
    } /* end of par2 loop over collision partners */

  } /* end of lspec loop over line producing species */


  /* Calculate the cumulatives (needed for indexing, see definitions.h) */

  for (lspec=0; lspec<nline_species; lspec++){

    for (par3=1; par3<ncolpar[lspec]; par3++){

      cum_ncoltran[SPECPAR(lspec,par3)] = cum_ncoltran[SPECPAR(lspec,par3-1)]
                                             + ncoltran[SPECPAR(lspec,par3-1)];

      cum_ncoltemp[SPECPAR(lspec,par3)] = cum_ncoltemp[SPECPAR(lspec,par3-1)]
                                             + ncoltemp[SPECPAR(lspec,par3-1)];

      cum_ncoltrantemp[SPECPAR(lspec,par3)] = cum_ncoltrantemp[SPECPAR(lspec,par3-1)]
                                                 + ( ncoltran[SPECPAR(lspec,par3-1)]
                                                     *ncoltemp[SPECPAR(lspec,par3-1)] );
/*
      printf("(3D-RT): cum_ncoltran[%d] = %d \n", par3, cum_ncoltran[SPECPAR(lspec,par3)]);
      printf("(3D-RT): cum_ncoltemp[%d] = %d \n", par3, cum_ncoltemp[SPECPAR(lspec,par3)]);
      printf( "(3D-RT): cum_ncoltrantemp[%d] = %d \n",
              par3, cum_ncoltrantemp[SPECPAR(lspec,par3)] );
*/
    }
  }


  for (lspec=0; lspec<nline_species; lspec++){

    tot_ncoltran[lspec] = cum_ncoltran[SPECPAR(lspec,ncolpar[lspec]-1)]
                          + ncoltran[SPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltemp[lspec] = cum_ncoltemp[SPECPAR(lspec,ncolpar[lspec]-1)]
                           + ncoltemp[SPECPAR(lspec,ncolpar[lspec]-1)];

    tot_ncoltrantemp[lspec] = cum_ncoltrantemp[SPECPAR(lspec,ncolpar[lspec]-1)]
                              + ( ncoltran[SPECPAR(lspec,ncolpar[lspec]-1)]
         	                  		  *ncoltemp[SPECPAR(lspec,ncolpar[lspec]-1)] );
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

  for (lspec=1; lspec<nline_species; lspec++){

    cum_tot_ncoltran[lspec] = cum_tot_ncoltran[lspec-1] + tot_ncoltran[lspec-1];

    cum_tot_ncoltemp[lspec] = cum_tot_ncoltemp[lspec-1] + tot_ncoltemp[lspec-1];

    cum_tot_ncoltrantemp[lspec] = cum_tot_ncoltrantemp[lspec-1] + tot_ncoltrantemp[lspec-1];
  }

  tot_cum_tot_ncoltran = cum_tot_ncoltran[nline_species-1] + tot_ncoltran[nline_species-1];
                                                        /* total over the line prodcing species */
  tot_cum_tot_ncoltemp = cum_tot_ncoltemp[nline_species-1] + tot_ncoltemp[nline_species-1];
                                                        /* total over the line prodcing species */
  tot_cum_tot_ncoltrantemp = cum_tot_ncoltrantemp[nline_species-1]
                                   + tot_ncoltrantemp[nline_species-1];
                                                        /* total over the line prodcing species */


  /* Define and allocate the collision related variables */

  double *coltemp;                                    /* Collision temperatures for each partner */
  coltemp = (double*) malloc( tot_cum_tot_ncoltemp*sizeof(double) );
                                                            /*[nline_species][ncolpar][ncoltemp] */

  double *C_data;                         /* C_data for each partner, transition and temperature */
  C_data = (double*) malloc( tot_cum_tot_ncoltrantemp*sizeof(double) );
                                                 /* [nline_species][ncolpar][ncoltran][ncoltemp] */

  int *icol;                   /* level index corresp. to collisional transition [0..ncoltran-1] */
  icol = (int*) malloc( tot_cum_tot_ncoltran*sizeof(int) );
                                                           /* [nline_species][ncolpar][ncoltran] */

  int *jcol;                   /* level index corresp. to collisional transition [0..ncoltran-1] */
  jcol = (int*) malloc( tot_cum_tot_ncoltran*sizeof(int) );
                                                           /* [nline_species][ncolpar][ncoltran] */

  // int *spec_par;                   /* number of the species corresponding to a collision partner */
  spec_par = (int*) malloc( tot_ncolpar*sizeof(int) );


  ortho_para = (char*) malloc( tot_ncolpar*sizeof(char) );


  for(int ind=0; ind<tot_ncolpar; ind++){

    spec_par[ind] = 0;

    ortho_para[ind] = 'i';
  }


  /* Initializing data */

  for(lspec=0; lspec<nline_species; lspec++){

    for (i=0; i<nlev[lspec]; i++){

      weight[SPECLEV(lspec,i)] = 0.0;
      energy[SPECLEV(lspec,i)] = 0.0;

      for (j=0; j<nlev[lspec]; j++){

	A_coeff[SPECLEVLEV(lspec,i,j)] = 0.0;
	B_coeff[SPECLEVLEV(lspec,i,j)] = 0.0;
	C_coeff[SPECLEVLEV(lspec,i,j)] = 0.0;

	frequency[SPECLEVLEV(lspec,i,j)] = 0.0;

	for (n=0; n<ngrid; n++){

	  R[SPECGRIDLEVLEV(lspec,n,i,j)] = 0.0;
	}
      }
    }
  }


  for(lspec=0; lspec<nline_species; lspec++){

    for (n=0; n<ngrid; n++){

      for (i=0; i<nlev[lspec]; i++){

        for (j=0; j<nlev[lspec]; j++){

	  R[SPECGRIDLEVLEV(lspec,n,i,j)] = 0.0;
	}
      }
    }
  }

   for(lspec=0; lspec<nline_species; lspec++){

    for (kr=0; kr<nrad[lspec]; kr++){

      irad[SPECRAD(lspec,kr)] = 0;
      jrad[SPECRAD(lspec,kr)] = 0;
    }
  }


  void read_linedata( string datafile, int *irad, int *jrad, double *energy, double *weight,
                      double *frequency, double *A_coeff, double *B_coeff, double *coltemp,
                      double *C_data, int *icol, int *jcol, int lspec );


  for(lspec=0; lspec<nline_species; lspec++){

    read_linedata( datafile[lspec], irad, jrad, energy, weight, frequency,
                   A_coeff, B_coeff, coltemp, C_data, icol, jcol, lspec );
  }



  // for(int ind=0; ind<ncolpar[0]; ind++){

  //   cout << "spec_par[" << ind << "] = " << spec_par[SPECPAR(0,ind)] << " (o/p?)" << ortho_para[SPECPAR(0,ind)] << " \n" ;
  // }



  /* Initializing populations */

  for (lspec=0; lspec<nline_species; lspec++){

    for (n=0; n<ngrid; n++){

      for (i=0; i<nlev[lspec]; i++){

        pop[SPECGRIDLEV(lspec,n,i)] = exp(-HH*CC*energy[SPECLEV(lspec,i)]/(KB*temperature[n]));

        if(n==5){printf("pop %lE energy %lE \n", pop[SPECGRIDLEV(lspec,n,i)], energy[SPECLEV(lspec,i)] );}
      }
    }
  }



  double *P_intensity;                                   /* Feautrier's mean intensity for a ray */
  P_intensity = (double*) malloc( ngrid*NRAYS*sizeof(double) );


  for (n1=0; n1<ngrid; n1++){

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

  for (lspec=0; lspec<nline_species; lspec++){

    level_populations( antipod, gridpoint, evalpoint, irad, jrad, frequency,
                       A_coeff, B_coeff, C_coeff, P_intensity, R, pop, dpop, C_data,
                       coltemp, icol, jcol, temperature, weight, energy, lspec, sobolev );
  }

  time_lp += omp_get_wtime();


  printf("\n(3D-RT): time in level_populations: %lf sec\n", time_lp);
  printf("(3D-RT): transfer done\n\n");

  printf("(3D-RT): Calculate column densities\n");
  

  printf("(3D-RT): Column densities calculated\n");



  double *column_density;               /* column densities for each species, ray and grid point */
  column_density = (double*) malloc( ngrid*nspec*NRAYS*sizeof(double) );

  double *rad_surface;
  rad_surface = (double*) malloc( ngrid*NRAYS*sizeof(double) );

  double *AV;                                   /* Visual extinction (only takes into account H) */
  AV = (double*) malloc( ngrid*NRAYS*sizeof(double) );

  metallicity = 1.0;

  double *UV_field;
  UV_field = (double*) malloc( ngrid*NRAYS*sizeof(double) );


  /* Initialization */

  for (n=0; n<ngrid; n++){

    for (r=0; r<NRAYS; r++){

      UV_field[RINDEX(n,r)]    = 0.0;
      rad_surface[RINDEX(n,r)] = 0.0;

      for (spec=0; spec<nspec; spec++){

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



  /* Free the allocated memory for temporary variables */

  free( unit_healpixvector );
  free( antipod );
  free( gridpoint );
  free( evalpoint );
  free( cum_raytot );
  free( key );
  free( raytot );
  free( P_intensity );
  free( species );
  free( spec_par );
  free( ortho_para );
  free( temperature );
  free( nlev );
  free( cum_nlev );
  free( cum_nlev2 );
  free( nrad );
  free( cum_nrad );
  free( irad );
  free( jrad );
  free( energy );
  free( weight );
  free( frequency );
  free( A_coeff );
  free( B_coeff );
  free( C_coeff );
  free( R );
  free( pop );
  free( dpop );
  free( ncolpar );
  free( cum_ncolpar );
  free( ncoltran );
  free( cum_ncoltran );
  free( tot_ncoltran );
  free( cum_tot_ncoltran );
  free( ncoltemp );
  free( cum_ncoltemp );
  free( tot_ncoltemp );
  free( cum_tot_ncoltemp );
  free( cum_ncoltrantemp );
  free( tot_ncoltrantemp );
  free( cum_tot_ncoltrantemp );
  free( coltemp );
  free( C_data );
  free( icol );
  free( jcol );



  printf("(3D-RT): done\n");

}

/*-----------------------------------------------------------------------------------------------*/
