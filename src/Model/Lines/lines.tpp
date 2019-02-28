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
    set_emissivity_and_opacity (
	      const long p,
        const int  l           )
{

  const LineProducingSpecies lspec = lineProducingSpecies[l];


  for (int k = 0; k < lspec.linedata.nrad; k++)
  {
    const long i = lspec.linedata.irad[k];
    const long j = lspec.linedata.jrad[k];

    const long ind   = index (p,l,k);
    const long ind_i = lspec.index (p,i);
    const long ind_j = lspec.index (p,j);

    emissivity[ind] = HH_OVER_FOUR_PI * lspec.linedata.A[k] * lspec.population(ind_i);

       opacity[ind] = HH_OVER_FOUR_PI * (  lspec.population(ind_j) * lspec.linedata.Ba[k]
                                          - lspec.population(ind_i) * lspec.linedata.Bs[k] );
  }


}
