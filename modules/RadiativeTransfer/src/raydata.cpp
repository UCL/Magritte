// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "raydata.hpp"
#include "GridTypes.hpp"
#include "types.hpp"


///  Constructor for RAYDATA
////////////////////////////

RAYDATA ::
RAYDATA (const long num_of_cells,
         const long num_of_freq_red,
         const long ray_nr,
         const long Ray_nr,
         const vReal2 &U_local,
         const vReal2 &V_local,
         const vReal3 &Ibdy_local,
         const Long1  &Cell2boundary_nr)
  : ncells             (num_of_cells)
  , nfreq_red          (num_of_freq_red)
  , ray                (ray_nr)
  , Ray                (Ray_nr)
  , U                  (U_local)
  , V                  (V_local)
  , boundary_intensity (Ibdy_local)
  , cell2boundary_nr   (Cell2boundary_nr)
{

  cellNrs.resize(ncells);
    notch.resize(ncells);
   lnotch.resize(ncells);
   shifts.resize(ncells);
      dZs.resize(ncells);


}   // END OF CONSTRUCTOR
