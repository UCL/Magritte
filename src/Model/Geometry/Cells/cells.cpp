// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "cells.hpp"
#include "Tools/constants.hpp"
#include "Tools/logger.hpp"


const string Cells::prefix = "Geometry/Cells/";


///  Reader for the Cells data from the Io object
///    @param[in] io         : Io object to read from
///    @param[in] parameters : Parameters object of the model
/////////////////////////////////////////////////////////////

int Cells ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  cout << "Reading cells" << endl;


  io.read_length (prefix+"cells", ncells);

  parameters.set_ncells (ncells);


  // Read cell centers and velocities
  x.resize (ncells);
  y.resize (ncells);
  z.resize (ncells);

  io.read_3_vector (prefix+"cells", x, y, z);

  vx.resize (ncells);
  vy.resize (ncells);
  vz.resize (ncells);

  io.read_3_vector (prefix+"velocities", vx, vy, vz);


  // Read number of neighbors
  n_neighbors.resize (ncells);

  io.read_list (prefix+"n_neighbors", n_neighbors);


  // Resize the neighbors to appropriate sizes
  neighbors.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    neighbors[p].resize (n_neighbors[p]);
  }


  // Read nearest neighbors lists
  io.read_array (prefix+"neighbors", neighbors);


  return (0);

}




///  Writer for the Cells data to the Io object
///  @param[in] io : io object
///////////////////////////////////////////////

int Cells ::
    write (
        const Io &io) const
{

  cout << "Writing cells" << endl;


  // Write cell centers and velocities

  io.write_3_vector (prefix+"cells", x, y, z);

  io.write_3_vector (prefix+"velocities", vx, vy, vz);


  // Write number of neighbors and neighbors lists

  io.write_list  (prefix+"n_neighbors", n_neighbors);

  io.write_array (prefix+"neighbors", neighbors);


  return (0);

}
