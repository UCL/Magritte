// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYDATA_HPP_INCLUDED__
#define __RAYDATA_HPP_INCLUDED__


#include <vector>


///  ProjectedCellData: data structure for the data projected on a ray
//////////////////////////////////////////////////////////////////////

struct ProjectedCellData
{

  long  cellNr;
  double shift;
  double    dZ;

  // Helper variables.
  long lnotch;
  long  notch;

};


/// RayData: projected cell data along a ray
////////////////////////////////////////////

typedef std::vector <ProjectedCellData> RayData;


#endif // __RAYDATA_HPP_INCLUDED__
