// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <limits>
using namespace std;

#include "cells.hpp"
#include "constants.hpp"


///  read: read the input into the data structure
///  @paran[in] io: io object
/////////////////////////////////////////////////

int Cells ::
    read (
        const Io &io)
{

  io.read_length ("cells/cells", ncells);


  // Read cell centers and velocities
  x.resize (ncells);
  y.resize (ncells);
  z.resize (ncells);

  io.read_3_vector ("cells/cells", x, y, z);

  vx.resize (ncells);
  vy.resize (ncells);
  vz.resize (ncells);

  io.read_3_vector ("cells/velocities", vx, vy, vz);


  // Convert velocities in m/s to fractions for C
  for (long p = 0; p < ncells; p++)
  {
    vx[p] /= CC;
    vy[p] /= CC;
    vz[p] /= CC;
  }


  // Read number of neighbors
  n_neighbors.resize (ncells);

  io.read_list ("cells/n_neighbors", n_neighbors);


  // Resize the neighbors to appropriate sizes
  neighbors.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    neighbors[p].resize (n_neighbors[p]);
  }


  // Read nearest neighbors lists
  io.read_array ("cells/neighbors", neighbors);


  // Resize boundary
          boundary.resize (ncells);
  boundary2cell_nr.resize (ncells);
  cell2boundary_nr.resize (ncells);


  // Initialise
  for (long p = 0; p < ncells; p++)
  {
            boundary[p] = false;
    cell2boundary_nr[p] = ncells;
    boundary2cell_nr[p] = ncells;
  }


  // Read boundary list
  io.read_list ("cells/boundary", boundary2cell_nr);


  // Set helper variables to identify the boundary
  for (long b = 0; b < nboundary; b++)
  {
    const long cell_nr = boundary2cell_nr[b];

    cell2boundary_nr[cell_nr] = b;
            boundary[cell_nr] = true;
  }


  return (0);

}




///  write: write the dat astructure
///  @paran[in] io: io object
////////////////////////////////////////////////

int Cells ::
    write (
        const Io &io) const
{

  // Write cell centers and velocities

  io.write_3_vector ("cells/cells", x, y, z);

  io.write_3_vector ("cells/velocities", vx, vy, vz);


  // Write number of neighbors and neighbors lists

  io.write_list ("cells/n_neighbors", n_neighbors);

  io.write_array ("cells/neighbors", neighbors);


  // Write boundary list

  io.write_list ("cells/boundary", boundary2cell_nr);


  return (0);

}
