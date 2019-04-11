// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/constants.hpp"
#include "Tools/types.hpp"
#include "Tools/debug.hpp"


///  index:
///////////

inline long Lines ::
    index (
        const long p,
        const int  l,
        const int  k) const
{
  return k + nrad_cum[l] + p * nlines;
}



///  index:
///////////

inline long Lines ::
    index (
        const long p,
        const long line_index) const
{
  return line_index + p * nlines;
}




///  set_emissivity_and_opacity
///    @param[in] p: number of cell
///    @param[in] l: number of line producing species
/////////////////////////////////////////////////////

inline void Lines ::
    set_emissivity_and_opacity ()
{


  OMP_PARALLEL_FOR (p, ncells)
  {
    for (int l = 0; l < nlspecs; l++)
    {
      for (int k = 0; k < lineProducingSpecies[l].linedata.nrad; k++)
      {
        const long ind = index (p,l,k);
        
        emissivity[ind] = lineProducingSpecies[l].get_emissivity (p, k);
           opacity[ind] = lineProducingSpecies[l].get_opacity    (p, k);
      }
    }
  }


}
