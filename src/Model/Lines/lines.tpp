// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/constants.hpp"
#include "Tools/types.hpp"
#include "Tools/debug.hpp"


///  Indexer for cell, line producing species and transition indices
///    @param[in] p : index of the cell
///    @param[in] l : index of the line producing species
///    @param[in] k : index of the line transition
////////////////////////////////////////////////////////////////////

inline long Lines :: index (const long p, const int l, const int k) const
{
  return k + nrad_cum[l] + p * nlines;
}



///  Indexer for cell and line indices
///    @param[in] p          : index of the cell
///    @param[in] line_index : index of the line
////////////////////////////////////////////////

inline long Lines :: index (const long p, const long line_index) const
{
  return line_index + p * nlines;
}




///  Setter for line emissivity and opacity
///    @param[in] p : index of the cell
///    @param[in] l : index of the line producing species
/////////////////////////////////////////////////////////

inline void Lines :: set_emissivity_and_opacity ()
{
    OMP_PARALLEL_FOR (p, ncells)
    {
        for (size_t l = 0; l < nlspecs; l++)
        {
            for (size_t k = 0; k < lineProducingSpecies[l].linedata.nrad; k++)
            {
                const size_t ind = index (p,l,k);

                emissivity[ind] = lineProducingSpecies[l].get_emissivity (p, k);
                   opacity[ind] = lineProducingSpecies[l].get_opacity    (p, k);
            }
        }
    }

}
