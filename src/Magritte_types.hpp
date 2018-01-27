// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_TYPES_HPP_INCLUDED__
#define __MAGRITTE_TYPES_HPP_INCLUDED__


// #include "Eigen/Dense"
#include <omp.h>




struct TEMPERATURE
{

  double dust;
  double gas;
  double gas_prev;

};

struct RAY
{
  double intensity;
};




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


    void new_column (long ncells)
    {

      double *H2 = new double[ncells*NRAYS];   // H2 column density for each ray and cell
      double *HD = new double[ncells*NRAYS];   // HD column density for each ray and cell
      double *C  = new double[ncells*NRAYS];   // C  column density for each ray and cell
      double *CO = new double[ncells*NRAYS];   // CO column density for each ray and cell

    }

# endif

};




struct CELL
{

  double x, y, z;            // x, y and z coordinate of cell center

  double vx, vy, vz;         // x, y and z component of velocity field

  long endpoint[NRAYS];      // cell numbers of ray endings
  double Z[NRAYS];           // distance from cell to boundary

  long neighbor[NRAYS];      // cell numbers of neighors
  long n_neighbors;          // number of neighbors


  double density;            // density

  double abundance[NSPEC];   // abundance for each species

  double rate[NREAC];        // reaction rate for each reaciton

  TEMPERATURE temperature;   // temperatures

  RAY ray[NRAYS];

  long id;                   // cell nr of associated cell in other grid
  bool removed;              // true when cell is removed

  bool boundary;             // true if boundary cell

};


struct EVALPOINT
{

  bool   onray;   // true when cell is on any ray thus an evaluation point

  long   ray;     // number of ray, evaluation point belongs to
  long   nr;      // number of evaluation point along ray

  double dZ;      // distance increment along ray
  double Z;       // distance between evaluation point and origin

  double vol;     // velocity along ray between grid point and evaluation point
  double dvc;     // velocity increment to next point in velocity space

  long next_in_velo;   // next point in velocity space

};


struct SPECIES
{

  std::string sym;            // chemical symbol

  double mass;                // molecular mass

  double initial_abundance;   // abundance before chemical evolution

};



struct LINE_SPECIES
{

  int nr[NLSPEC];                // symbol of line producing species

  std::string sym[NLSPEC];       // symbol of line producing species


  int irad[TOT_NRAD];            // level index of radiative transition
  int jrad[TOT_NRAD];            // level index of radiative transition

  double energy[TOT_NLEV];       // energy of level
  double weight[TOT_NLEV];       // statistical weight of level

  double frequency[TOT_NLEV2];   // frequency corresponing to i -> j transition

  double A_coeff[TOT_NLEV2];     // Einstein A_ij coefficient
  double B_coeff[TOT_NLEV2];     // Einstein B_ij coefficient


  // Collision related variables

  int partner[TOT_NCOLPAR];                  // number of species corresponding to a collision partner

  char ortho_para[TOT_NCOLPAR];              // stores whether it is ortho or para H2

  double coltemp[TOT_CUM_TOT_NCOLTEMP];      // Collision temperatures for each partner

  double C_data[TOT_CUM_TOT_NCOLTRANTEMP];   // C_data for each partner, tran. and temp.

  int icol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition
  int jcol[TOT_CUM_TOT_NCOLTRAN];            // level index corresp. to col. transition

};

// struct COLPAR
// {
//
//   int nr;            // nr of species of collision partner
//
//   char ortho_para;   // o when ortho, p when para and n when NA
//
// };





// struct LINE_SPECIES
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




struct REACTION
{

  std::string R1;   // reactant 1
  std::string R2;   // reactant 2
  std::string R3;   // reactant 3

  std::string P1;   // reaction product 1
  std::string P2;   // reaction product 2
  std::string P3;   // reaction product 3
  std::string P4;   // reaction product 4


  double alpha;     // alpha coefficient to calculate rate coefficient k
  double beta;      // beta  coefficient to calculate rate coefficient k
  double gamma;     // gamma coefficient to calculate rate coefficient k

  double RT_min;    // RT_min coefficient to calculate rate coefficient k
  double RT_max;    // RT_max coefficient to calculate rate coefficient k

  int    dup;       // Number of duplicates of this reaction

};




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





#endif //__MAGRITTE_TYPES_HPP_INCLUDED__
