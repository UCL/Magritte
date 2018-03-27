// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_TYPES_HPP_INCLUDED__
#define __MAGRITTE_TYPES_HPP_INCLUDED__


// #include "Eigen/Dense"
#include <omp.h>

// Include CELLS class
#include "cells.hpp"
// Include RAYS class
#include "healpixvectors.hpp"
// Include SPECIES class
#include "species.hpp"
// Include REACTIONS class
#include "reactions.hpp"
// Include LINES class
#include "lines.hpp"



// struct TEMPERATURE
// {
//   double dust;       // dust temperature
//   double gas;        // gas temperature
//   double gas_prev;   // gas temperature in previous iteration
//
// };
//
//
// struct RAY
// {
//   double intensity;
//   double column;
//
//   double rad_surface;
//   double AV;
// };




struct COLUMN_DENSITIES
{

# if (FIXED_NCELLS)

    double H2[NCELLS*NRAYS];    // H2 column density
    double HD[NCELLS*NRAYS];    // HD column density
    double C[NCELLS*NRAYS];     // C  column density
    double CO[NCELLS*NRAYS];    // CO column density

    double tot[NCELLS*NRAYS];   // total column density

# else

    double *H2;    // H2 column density
    double *HD;    // HD column density
    double *C;     // C  column density
    double *CO;    // CO column density

    double *tot;   // total column density


    void new_columns (long ncells)
    {
      H2  = new double[ncells*NRAYS];   // H2 column density for each ray and cell
      HD  = new double[ncells*NRAYS];   // HD column density for each ray and cell
      C   = new double[ncells*NRAYS];   // C  column density for each ray and cell
      CO  = new double[ncells*NRAYS];   // CO column density for each ray and cell

      tot = new double[ncells*NRAYS];   // CO column density for each ray and cell
    }

    void delete_columns ()
    {
      delete [] H2;
      delete [] HD;
      delete [] C ;
      delete [] CO;

      delete [] tot;
    }

# endif

};



// struct CELL   // (array of structures)
// {
//
//   // Standard input
//
//   double x,  y,  z;          // x, y and z coordinate of cell center
//   double vx, vy, vz;         // x, y and z component of velocity field
//
//   double density;            // total density in cell
//
//
//   // Geometry
//
//   long endpoint[NRAYS];      // cell numbers of ray endings
//   double Z[NRAYS];           // distance from cell to boundary
//
//   long neighbor[NRAYS];      // cell numbers of neighors
//   long n_neighbors;          // number of neighbors
//
//
//   // Chemistry
//
//   double abundance[NSPEC];   // abundance for each species
//   double rate[NREAC];        // reaction rate for each reaciton
//
//   // Lines
//
//   double pop[TOT_NLEV];              // level population
//   double mean_intensity[TOT_NRAD];   // mean intensity
//
//   TEMPERATURE temperature;   // temperatures
//
//   RAY ray[NRAYS];
//
//   double UV;                 // average UV intensity
//
//   long id;                   // cell nr of associated cell in other grid
//   bool removed;              // true when cell is removed
//
//   bool boundary;             // true if boundary cell
//   bool mirror;               // true if reflective boundary
//
//   double thermal_ratio;
//   double thermal_ratio_prev;
//
// };




// struct SPECIES
// {
//
//   std::string sym;            // chemical symbol
//
//   double mass;                // molecular mass
//
//   double initial_abundance;   // abundance before chemical evolution
//
// };




// struct LINES   // (structure of arrays)
// {
//
//   int nr[NLSPEC];                // symbol of line producing species
//   std::string sym[NLSPEC];       // symbol of line producing species
//
//
//   int irad[TOT_NRAD];            // level index of radiative transition
//   int jrad[TOT_NRAD];            // level index of radiative transition
//
//   double energy[TOT_NLEV];       // energy of level
//   double weight[TOT_NLEV];       // statistical weight of level
//
//   double frequency[TOT_NLEV2];   // frequency corresponing to i -> j transition
//
//   double A_coeff[TOT_NLEV2];     // Einstein A_ij coefficient
//   double B_coeff[TOT_NLEV2];     // Einstein B_ij coefficient
//
//
//   // Collision related variables
//
//   int partner[TOT_NCOLPAR];                  // species number corresponding to a collision partner
//
//   char ortho_para[TOT_NCOLPAR];              // stores whether it is ortho or para H2
//
//   double coltemp[TOT_CUM_TOT_NCOLTEMP];      // Collision temperatures for each partner
//
//   double C_data[TOT_CUM_TOT_NCOLTRANTEMP];   // C_data for each partner, tran. and temp.
//
//   int icol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition
//   int jcol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition
//
// };

// struct COLPAR
// {
//
//   int nr;            // nr of species of collision partner
//
//   char ortho_para;   // o when ortho, p when para and n when NA
//
// };





// struct LINES
// {
//
//   int nr;                      // nr of corresponding species
//
//   std::string sym;             // chemical symbol
//
//
//   int irad[MAX_NRAD];
//   int jrad[MAX_NRAD];
//
//   double A[MAX_NLEV][MAX_NLEV];
//   double B[MAX_NLEV][MAX_NLEV];
//   double C[MAX_NLEV][MAX_NLEV];
//
//   // Eigen::MatrixXd EA(MAX_NLEV, MAX_NLEV);
//   // Eigen::MatrixXd EC(MAX_NLEV, MAX_NLEV);
//   // Eigen::MatrixXd EB(MAX_NLEV, MAX_NLEV);
//
//   // ERROR -> read up on constructors...
//
//
//   double frequency[MAX_NLEV][MAX_NLEV];
//
//   double energy[MAX_NLEV];
//   double weight[MAX_NLEV];
//
//   // COLPAR colpar[MAX_NCOLPAR];
//
//   // int spec_par[MAX_NCOLPAR];
//
//   // char ortho_para[MAX_NCOLPAR];
//
//   // int icol[TOT_NCOLTRAN];
//   // int jcol[TOT_NCOLTRAN];
//   //
//   // double coltemp[MAX_TOT_NCOLTEMP];
//   //
//   // double C_data[MAX_TOT_NCOLTRANTEMP];
//
// };



//
// struct REACTION
// {
//
//   std::string R1;   // reactant 1
//   std::string R2;   // reactant 2
//   std::string R3;   // reactant 3
//
//   std::string P1;   // reaction product 1
//   std::string P2;   // reaction product 2
//   std::string P3;   // reaction product 3
//   std::string P4;   // reaction product 4
//
//
//   double alpha;     // alpha coefficient to calculate rate coefficient k
//   double beta;      // beta  coefficient to calculate rate coefficient k
//   double gamma;     // gamma coefficient to calculate rate coefficient k
//
//   double RT_min;    // RT_min coefficient to calculate rate coefficient k
//   double RT_max;    // RT_max coefficient to calculate rate coefficient k
//
//   int    dup;       // Number of duplicates of this reaction
//
// };




struct TIMER
{

  double duration;


  void initialize()
  {
    duration = 0.0;
  }


  void start()
  {
    duration -= omp_get_wtime();
  }


  void stop()
  {
    duration += omp_get_wtime();
  }

};




struct TIMERS
{

  TIMER total;
  TIMER chemistry;
  TIMER level_pop;


  void initialize()
  {
    total.initialize();
    chemistry.initialize();
    level_pop.initialize();
  }

};



struct NITERATIONS
{

  int tb;
  int rt;

  void initialize()
  {
    tb = 0;
    rt = 0;
  }

};

// Include LINES class
// #include "lines.hpp"



#endif //__MAGRITTE_TYPES_HPP_INCLUDED__
