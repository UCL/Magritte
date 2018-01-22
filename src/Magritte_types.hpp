// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_TYPES_HPP_INCLUDED__
#define __MAGRITTE_TYPES_HPP_INCLUDED__


#include "Eigen/Dense"


struct TEMPERATURE
{

  double dust;
  double gas;
  double gas_prev;

};


struct COLUMN_DENSITIES
{

  double H2;    // H2 column density
  double HD;    // HD column density
  double C;     // C  column density
  double CO;    // CO column density

  double tot;   // total column density

};


struct RAY
{

  double intensity;

  COLUMN_DENSITIES column;

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

  RAY ray[NRAYS];            // discretized directions

  long id;                   // cell nr of associated cell in other grid
  bool removed;              // true when cell is removed

  bool boundary;             // true if boundary cell

};



// struct CELL
// {
//
//   double x, y, z;            // x, y and z coordinate of cell center
//
//   long endpoint[NRAYS];      // cell numbers of ray endings
//   double Z[NRAYS];           // distance from cell to boundary
//
//   long neighbor[NRAYS];      // cell numbers of neighors
//   long n_neighbors;          // number of neighbors
//
//   double vx, vy, vz;         // x, y and z component of velocity field
//
//   double density;            // density
//
//   double abundance[NSPEC];   // abundance for each species
//
//   double rate[NREAC];        // reaction rate for each reaciton
//
//   TEMPERATURE temperature;   // temperatures
//
//   RAY ray[NRAYS];            // discretized directions
//
//   long id;                   // cell nr of associated cell in other grid
//   bool removed;              // true when cell is removed
//
//   bool boundary;             // true if boundary cell
//
// };




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


struct COLPAR
{

  int nr;            // nr of species of collision partner

  char ortho_para;   // o when ortho, p when para and n when NA

};


struct LINE_SPECIES
{

  int nr;                      // nr of corresponding species

  std::string sym;             // chemical symbol


  int irad[MAX_NRAD];
  int jrad[MAX_NRAD];

  double A[MAX_NLEV][MAX_NLEV];
  double B[MAX_NLEV][MAX_NLEV];
  double C[MAX_NLEV][MAX_NLEV];

  // Eigen::MatrixXd EA(MAX_NLEV, MAX_NLEV);
  // Eigen::MatrixXd EC(MAX_NLEV, MAX_NLEV);
  // Eigen::MatrixXd EB(MAX_NLEV, MAX_NLEV);

  // ERROR -> read up on constructors...


  double frequency[MAX_NLEV][MAX_NLEV];

  double energy[MAX_NLEV];
  double weight[MAX_NLEV];

  // COLPAR colpar[MAX_NCOLPAR];

  // int spec_par[MAX_NCOLPAR];

  // char ortho_para[MAX_NCOLPAR];

  // int icol[TOT_NCOLTRAN];
  // int jcol[TOT_NCOLTRAN];
  //
  // double coltemp[MAX_TOT_NCOLTEMP];
  //
  // double C_data[MAX_TOT_NCOLTRANTEMP];

};


// typedef struct
// {
//
//
//
//
// } ALL_LINE_SPECIES




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


#endif //__MAGRITTE_TYPES_HPP_INCLUDED__
