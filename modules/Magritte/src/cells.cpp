// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <limits>
using namespace std;

#include "cells.hpp"
#include "constants.hpp"


///  Constructor for Cells:
///    @param[in] io: io object
/////////////////////////////////////

Cells ::
    Cells (
        const Io &io)
  : ncells    (io.get_length ("cells")),
    nrays     (io.get_length ("rays")),
    nboundary (io.get_length ("boundary")),
    rays      (io)
{

  cout << "ncells!!!!!!!!!!!" << endl;
  cout << ncells << endl;

  allocate ();

  initialise ();

  read (io);

  setup ();


}   // END OF CONSTRUCTOR




///  allocate: resize all data structures
/////////////////////////////////////////

int Cells ::
    allocate ()
{

  x.resize (ncells);
  y.resize (ncells);
  z.resize (ncells);

  vx.resize (ncells);
  vy.resize (ncells);
  vz.resize (ncells);

  n_neighbors.resize (ncells);
    neighbors.resize (ncells);

  boundary.resize (ncells);
  // mirror.resize (ncells);

  boundary2cell_nr.resize (ncells);
  cell2boundary_nr.resize (ncells);


  return (0);

}




///  initialise: initialise all data structures
///////////////////////////////////////////////

int Cells ::
    initialise ()
{

  for (long p = 0; p < ncells; p++)
  {
    boundary[p] = false;
    //mirror[p]   = false;

    cell2boundary_nr[p] = ncells;
    boundary2cell_nr[p] = ncells;
  }


  return (0);

}




///  read: read the input into the data structure
///  @paran[in] io: io object
/////////////////////////////////////////////////

int Cells ::
    read (
        const Io &io)
{

  // Read cell centers and velocities

  io.read_3_vector ("cells", x, y, z);

  io.read_3_vector ("velocities", vx, vy, vz);


  // Read number of neighbors

  io.read_list ("n_neighbors", n_neighbors);


  // Resize the neighbors to appropriate sizes

  for (long p = 0; p < ncells; p++)
  {
    neighbors[p].resize (n_neighbors[p]);
  }


  // Read nearest neighbors lists

  io.read_array ("neighbors", neighbors);


  // Read boundary list

  io.read_list ("boundary", boundary2cell_nr);


  return (0);

}




///  setup: setup data structure
////////////////////////////////

int Cells ::
    setup ()
{

  // Convert velocities in m/s to fractions for C

  for (long p = 0; p < ncells; p++)
  {
    vx[p] /= CC;
    vy[p] /= CC;
    vz[p] /= CC;
  }


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

  io.write_3_vector ("cells", x, y, z);

  io.write_3_vector ("velocities", vx, vy, vz);


  // Write number of neighbors and neighbors lists

  io.write_list ("n_neighbors", n_neighbors);

  io.write_array ("neighbors", neighbors);


  // Write boundary list

  io.write_list ("boundary", boundary2cell_nr);


  return (0);

}
