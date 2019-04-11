// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "cells.hpp"
#include "Tools/constants.hpp"
#include "Tools/logger.hpp"


const string Cells::prefix = "Geometry/Cells/";


///  read: read the input into the data structure
///    @paran[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Cells ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  write_to_log ("Reading cells");


  io.read_length (prefix+"cells", ncells);

  write_to_log("ncells = ", ncells);


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


  // Convert velocities in m/s to fractions for C
  for (long p = 0; p < ncells; p++)
  {
    vx[p] /= CC;
    vy[p] /= CC;
    vz[p] /= CC;
  }


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




///  write: write the dat astructure
///  @paran[in] io: io object
////////////////////////////////////////////////

int Cells ::
    write (
        const Io &io) const
{

  write_to_log ("Writing cells");


  // Write cell centers and velocities

  io.write_3_vector (prefix+"cells", x, y, z);

  write_to_log ("Succes in writing coordinates");

  io.write_3_vector (prefix+"velocities", vx, vy, vz);

  write_to_log ("Succes in writing velocities");

  // Write number of neighbors and neighbors lists

  io.write_list  (prefix+"n_neighbors", n_neighbors);

  io.write_array (prefix+"neighbors", neighbors);

  write_to_log ("Succes in writing cells");


  return (0);

}