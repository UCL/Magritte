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
///    @param[in] io: io data object
////////////////////////////////////

Model ::
    Model ()
{


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




///  read: read model data
///    @param[in] io: io data object
////////////////////////////////////

int Model ::
   read (
      const Io &io)
{

        cells.read (io);

  temperature.read (io);

      species.read (io);


  // Get nlespecs
  io.read_length ("linedata", nlspecs);

  for (int l = 0; l < nlspecs; l++)
  {
    linedata[l].read (io, l);
  }


 return (0);

}




///  write: write out model data
///    @param[in] io: io data object
////////////////////////////////////

int Model ::
   write (
      const Io &io) const
{

        cells.write (io);

  temperature.write (io);

      species.write (io);


  for (int l = 0; l < nlspecs; l++)
  {
    linedata[l].write (io, l);
  }


 return (0);

}
