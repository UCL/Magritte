// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYPAIR_HPP_INCLUDED
#define __RAYPAIR_HPP_INCLUDED


#include "GridTypes.hpp"


///  RAYDATA: data structure for the data along a ray
/////////////////////////////////////////////////////

struct RAYPAIR
{

  const long ncells;      ///< number of cells
  
  RAYDATA ray_r;
  RAYDATA ray_ar;

  vReal   Su [ncells];   // effective source for u along ray r
  vReal   Sv [ncells];   // effective source for v along ray r
  vReal dtau [ncells];   // optical depth increment along ray r

  vReal Lambda [ncells];

  const long ndep = n_r + n_ar;


  // Extract the cell on ray r and antipodal ar

  long n = 0;
  

  RAYDATA (const long num_of_cells,
           const long origin_Ray   );

  int initialize (const RAYDATA &raydata_r,
                  const RAYDATA &raydata_ar);



};


#endif // __RAYPAIR_HPP_INCLUDED
