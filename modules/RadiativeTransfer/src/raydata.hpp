// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYDATA_HPP_INCLUDED
#define __RAYDATA_HPP_INCLUDED


#include "GridTypes.hpp"


///  RAYDATA: data structure for the data along a ray
/////////////////////////////////////////////////////

struct RAYDATA
{

  const long ncells;      ///< number of cells
  
  Long1  cellNrs [ncells];
  Long1    notch [ncells];
  Long1   lnotch [ncells];
  Double1 shifts [ncells];   // indicates where we are in frequency space
  Double1    dZs [ncells];


  // Extract the cell on ray r and antipodal ar

  long n = 0;
  

  RAYDATA (const long num_of_cells);

  int initialize (CELLS<Dimension, Nrays> &cells);



};



template <int Dimension, long Nrays>
inline int initialize (CELLS<Dimension,Nrays> &cells)
{

  // Reset number of cells on the ray

  n = 0;     


  // Find projected cells on ray

  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last increment in Z

  long nxt = cells.next (origin, ray, origin, Z, dZ);


  if (nxt != ncells)   // if we are not going out of grid
  {
    cellNrs[n] = nxt;
        dZs[n] = dZ;

    n++;

    while (!cells.boundary[nxt])   // while we have not hit the boundary
    {
      nxt = cells.next (origin, ray, nxt, Z, dZ);

      cellNrs[n] = nxt;
          dZs[n] = dZ;

      n++;
    }
  }


  // Initialize notches and shitfs

  for (long q = 0; q < n; q++)
  {
     notch[q] = 0;
    lnotch[q] = 0;
    shifts[q] = 1.0 - cells.relative_velocity (o, r, cellNrs[q]) / CC;
  }
  
  lnotch_r[ncells] = 0;


  return (0);
    
}


#endif // __RAYDATA_HPP_INCLUDED
