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

  const long ncells;          ///< number of cells
  const long nrays;           ///< number of rays
  const long nrays_red;       ///< reduced number of rays
  const long nfreq_red;       ///< reduced number of frequencies
  
  
  vReal2 Ip_out;              ///< intensity out along ray r  (r, index(p,f))
  vReal2 Im_out;              ///< intensity out along ray ar (r, index(p,f))
  
  
  IMAGE (const long num_of_cells,
         const long num_of_rays,
         const long num_of_freq_red);

  static long get_nrays_red (const long nrays);


  //int initialize ();

  int read (const string boundary_intensity_file);

  int write (const string boundary_intensity_file) const;

  inline long index (const long p,
                     const long f ) const;


  // Print

  int print (const string tag) const;


};


inline long IMAGE ::
            index (const long p, const long f) const
{
  return f + p*nfreq_red;
}


#endif // __IMAGE_HPP_INCLUDED
