// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "model.hpp"
#include "io.hpp"


///  Constructor for Model
///    @param[in] input: input data object
///////////////////////////////////////////

Model ::
    Model (
        const Io &io)
  : ncells      (io.get_length ("cells")),
    nrays       (io.get_length ("rays")),
    nspecs      (io.get_length ("species")),
    cells       (io),
    temperature (io),
    species     (io),
    linedata    (io)
{

//cout << input.get_length("cells") << endl;

}   // END OF CONSTRUCTOR



///// setup: set up the necesary sub data structures
//
//int Model ::
//    setup ()
//{
//
//    cells.setup ();
//
//  species.setup ();
//
//
//  return (0);
//
//}
//
//
//
//


/// write: write out model data
///////////////////////////////

int Model ::
   write (
      const Io &io) const
{

        cells.write (io);

  temperature.write (io);

      species.write (io);


 return (0);

}
