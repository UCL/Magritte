// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_TYPES_HPP_INCLUDED__
#define __MAGRITTE_TYPES_HPP_INCLUDED__


typedef struct
{
  double dust;
  double gas;
  double gas_prev;

} TEMPERATURE;




typedef struct
{
  double x, y, z;            // x, y and z coordinate of cell center

  long neighbor[NRAYS];      // cell numbers of neighors
  long n_neighbors;          // number of neighbors

  double vx, vy, vz;         // x, y and z component of velocity field

  double density;            // density

  double abundance[NSPEC];   // abundance for each species

  double rate[NREAC];        // reaction rate for each reaciton

  TEMPERATURE temperature;   // temperatures

  long id;                   // cell nr of associated cell in reduced grid

  bool removed;              // true when cell is removed

} CELL;




typedef struct
{
  bool   onray;   // true when cell is on any ray thus an evaluation point

  long   ray;     // number of ray, evaluation point belongs to
  long   nr;      // number of evaluation point along ray

  double dZ;      // distance increment along ray
  double Z;       // distance between evaluation point and origin

  double vol;     // velocity along ray between grid point and evaluation point
  double dvc;     // velocity increment to next point in velocity space

  long next_in_velo;   // next point in velocity space

} EVALPOINT;




typedef struct
{
  std::string sym;            // chemical symbol

  double mass;                // molecular mass

  double initial_abundance;   // abundance before chemical evolution

} SPECIES;




// typedef struct
// {
//
//   int spec;                    // nr of corresponding species
//
//   int irad[ntran];
//   int jrad[ntran];
//
//   double A[nlev*nlev];
//   double B[nlev*nlev];
//
//   double frequency[nlev*nlev];
//
//   double energy[nlev];
//   double weight[nlev];
//
//   int icol[ncoltran];
//   int jcol[ncoltran];
//
//
//
//
// } LINE_SPECIES;


// typedef struct
// {
//
//
//
// } ALL_LINE_SPECIES




typedef struct
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

} REACTION;


#endif //__MAGRITTE_TYPES_HPP_INCLUDED__
