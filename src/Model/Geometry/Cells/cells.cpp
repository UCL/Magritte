// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "cells.hpp"
#include "Tools/constants.hpp"
#include "Tools/logger.hpp"


const string Cells::prefix = "Geometry/Cells/";


///  Read the Cells data from the Io object
///    @param[in] io:         Io object to read from
///    @param[in] parameters: Parameters object of the model
////////////////////////////////////////////////////////////

int Cells ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  cout << "Reading cells" << endl;


  io.read_length (prefix+"cells", ncells_plus_ncameras);

  ncameras = parameters.ncameras ();
  ncells   = ncells_plus_ncameras - ncameras;

  cout << "ncells                = " << ncells << endl;
  cout << "ncameras              = " << ncameras << endl;
  cout << "ncells_plus_ncameras  = " << ncells_plus_ncameras << endl;


  parameters.set_ncells (ncells);


  // Read cell centers and velocities
  x.resize (ncells_plus_ncameras);
  y.resize (ncells_plus_ncameras);
  z.resize (ncells_plus_ncameras);

  io.read_3_vector (prefix+"cells", x, y, z);

  vx.resize (ncells_plus_ncameras);
  vy.resize (ncells_plus_ncameras);
  vz.resize (ncells_plus_ncameras);

  io.read_3_vector (prefix+"velocities", vx, vy, vz);


  // Convert velocities in m/s to fractions for C
  for (long p = 0; p < ncells_plus_ncameras; p++)
  {
    vx[p] /= CC;
    vy[p] /= CC;
    vz[p] /= CC;
  }


  // Read number of neighbors
  n_neighbors.resize (ncells_plus_ncameras);

  io.read_list (prefix+"n_neighbors", n_neighbors);


  // Resize the neighbors to appropriate sizes
  neighbors.resize (ncells_plus_ncameras);

  for (long p = 0; p < ncells_plus_ncameras; p++)
  {
    neighbors[p].resize (n_neighbors[p]);
  }


  // Read nearest neighbors lists
  io.read_array (prefix+"neighbors", neighbors);


  return (0);

}




///  write: write the dat astructure
///  @param[in] io: io object
////////////////////////////////////////////////

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
