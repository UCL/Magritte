// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "temperature.hpp"
#include "constants.hpp"
#include "types.hpp"


///  Constructor for Temperature
///    @param[in] io: io object
////////////////////////////////

Temperature ::
    Temperature (
        const Io &io)
    : ncells (io.get_length ("temperature"))
{

    allocate ();

    read (io);

    setup ();


}   // END OF CONSTRUCTOR




///  allocate: resize data structure
////////////////////////////////////

int Temperature ::
    allocate ()
{

       gas.resize (ncells);
      dust.resize (ncells);
  gas_prev.resize (ncells);

    vturb2.resize (ncells);


  return (0);

}




///  read: read in data structure
///    @param[in] io: io object
/////////////////////////////////

int Temperature ::
    read (
        const Io &io)
{

  // Read gas temperature file

  io.read_list ("temperature", gas);


 // Read gas turbulence file

  io.read_list ("vturbulence", vturb2);


  return (0);

}




///  setup: set up data structure
/////////////////////////////////

int Temperature ::
    setup ()
{

  // Convert to square of turbulent velocity w.r.t. c

  for (long p = 0; p < ncells; p++)
  {
    vturb2[p] /= CC;          // devide by speed of light
    vturb2[p] *= vturb2[p];   // square
  }


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
/////////////////////////////////

int Temperature ::
    write (
        const Io &io) const
{

  // Read gas temperature file

  io.write_list ("temperature", gas);


 // Read gas turbulence file

  io.write_list ("vturbulence", vturb2);


  return (0);

}
