// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __IMAGE_HPP_INCLUDED
#define __IMAGE_HPP_INCLUDED


#include "GridTypes.hpp"


///  IMAGE: data structure for the images
/////////////////////////////////////////

struct IMAGE
{

  const long ncells;      ///< number of cells
  const long nrays;       ///< number of rays
  const long nrays_red;   ///< reduced number of rays
  const long nfreq_red;   ///< reduced number of frequencies
    
  vReal3 I_p;             ///< intensity out along ray r  (r, index(p,f))
  vReal3 I_m;             ///< intensity out along ray r  (r, index(p,f))
  
  
  IMAGE (const long num_of_cells,
         const long num_of_rays,
         const long num_of_freq_red);


  int print (const string tag) const;


};


#endif // __IMAGE_HPP_INCLUDED
