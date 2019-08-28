// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "turbulence.hpp"
#include "Tools/constants.hpp"
#include "Tools/logger.hpp"


const string Turbulence::prefix = "Thermodynamics/Turbulence/";


///  read: read in data structure
///    @param[in] io: io object
///    @param[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Turbulence ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  cout << "Reading turbulence" << endl;


  // Get number of cells from length of temperature/gas file
  io.read_length (prefix+"vturb2", ncells);


  parameters.set_ncells (ncells);


  vturb2.resize (ncells);


  // Read gas turbulence file
  io.read_list (prefix+"vturb2", vturb2);


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
/////////////////////////////////

int Turbulence ::
    write (
        const Io &io) const
{

  cout << "Writing turbulence" << endl;


  // Read gas turbulence file
  io.write_list (prefix+"vturb2", vturb2);


  return (0);

}
