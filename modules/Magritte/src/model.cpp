// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
using namespace std;

#include "model.hpp"
#include "input.hpp"


///  Constructor for Model
///    @param[in] input_folder: folder containing input data
////////////////////////////////////////////////////////////

Model ::
    Model (
        const Input input)
  : ncells      (input.get_ncells()),
    nrays       (input.get_nrays ()),
    nfreqs      (input.get_nfreqs()),
    nspecs      (input.get_nspecs()),
    cells       (input),
    temperature (input),
    species     (input)
{


}   // END OF CONSTRUCTOR



/// setup: set up the necesary sub data structures

int Model ::
    setup ()
{

    cells.setup ();

  species.setup ();


  return (0);

}




/// read: read in the data

int Model ::
    read (const Input input)
{

        cells.read (input);

  temperature.read (input);


  return (0);

}
