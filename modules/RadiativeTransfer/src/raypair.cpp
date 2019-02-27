// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "raypair.hpp"
#include "GridTypes.hpp"


///  Constructor for RAYPAIR
////////////////////////////

RAYPAIR ::
RAYPAIR                     (
    const long num_of_cells,
    const long num_of_freq_red,
    const long ray_nr,
    const long aray_nr,
    const long Ray_nr,
    const vReal2 &U_local,
    const vReal2 &V_local,
    const vReal3 &Ibdy_local,
    const Long1  &Cell2boundary_nr)
  : ncells     (num_of_cells)
  , ray        (ray_nr)
  , aray       (aray_nr)
  , Ray        (Ray_nr)
  , raydata_r  (ncells, num_of_freq_red, ray,  Ray, U_local, V_local, Ibdy_local, Cell2boundary_nr)
  , raydata_ar (ncells, num_of_freq_red, aray, Ray, U_local, V_local, Ibdy_local, Cell2boundary_nr)
{

}   // END OF CONSTRUCTOR
