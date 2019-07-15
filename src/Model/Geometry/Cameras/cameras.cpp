// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "cameras.hpp"
#include "Tools/logger.hpp"


const string Cameras::prefix = "Geometry/Cameras/";


///  read: read the input into the data structure
///  @param[in] io: io object
///  @param[in] parameters: model parameters object
///////////////////////////////////////////////////

int Cameras ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  cout << "Reading cameras" << endl;


  // Read camera list
  io.read_length (prefix+"camera2cell_nr", ncameras);

  cout << "ncameras read is " << ncameras << endl;

  // Resize camera list
  camera2cell_nr.resize (ncameras);

  io.read_list   (prefix+"camera2cell_nr", camera2cell_nr);

  // Set model parameter
  parameters.set_ncameras (ncameras);


  return (0);

}




///  write: write the data structure
///  @param[in] io: io object
////////////////////////////////////////////////

int Cameras ::
    write (
        const Io &io) const
{

  cout << "Writing cameras" << endl;


  io.write_list (prefix+"camera2cell_nr", camera2cell_nr);


  return (0);

}
