// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYDATA_HPP_INCLUDED
#define __RAYDATA_HPP_INCLUDED




///  RAYDATA: data structure for the data along a ray
/////////////////////////////////////////////////////

struct RAYDATA
{

  const long ncells;      ///< number of cells
  const long origin;      ///< cell nr of origin
  const long ray;         ///< (global) index of ray
  const long Ray;         ///< (local) index of ray
  
  Long1  cellNrs [ncells];
  Long1    notch [ncells];
  Long1   lnotch [ncells];
  Double1 shifts [ncells];   // indicates where we are in frequency space
  Double1    dZs [ncells];

  vReal   chi_c;
  vReal term1_c;
  vReal term2_c;

  long n = 0;                // Number of (projected) cells on this ray
  

  RAYDATA (const long num_of_cells,
           const long origin_cell,
           const long origin_ray,
           const long origin_Ray   );

  int initialize (const CELLS<Dimension, Nrays> &cells);

  int get_dtau_Su_Sv (const FREQUENCIES &frequencies,
                      const TEMPERATURE &temperature,
                      const LINES       &lines,
                      const SCATTERING  &scattering,
                      const RADIATION   &radiation,
                      const long         f,
                      const long         q,
                            double      &dtau,
                            double      &Su,
                            double      &Sv         );


};


#include "raydata.tpp"


#endif // __RAYDATA_HPP_INCLUDED
