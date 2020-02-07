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

int Cells :: read (const Io &io, Parameters &parameters)
{
    cout << "Reading cells..." << endl;

    /// Read and set ncells
    io.read_length (prefix+"position", ncells);
    parameters.set_ncells (ncells);

    position.resize (ncells);
    velocity.resize (ncells);

    Double2 position_array (position.size(), Double1(3));
    Double2 velocity_array (velocity.size(), Double1(3));

    io.read_array(prefix+"position", position_array);
    io.read_array(prefix+"velocity", velocity_array);

    for (size_t p=0; p<position.size(); p++)
    {
        position[p] = {position_array[p][0], position_array[p][1], position_array[p][2]};
    }

    for (size_t p=0; p<velocity.size(); p++)
    {
        velocity[p] = {velocity_array[p][0], velocity_array[p][1], velocity_array[p][2]};
    }

    // Read cell centers and velocities


  // Read number of neighbors
  n_neighbors.resize (ncells);

  io.read_list (prefix+"n_neighbors", n_neighbors);


  const long max_n_neighbors = *std::max_element (n_neighbors.begin(),
                                                  n_neighbors.end  () );


  // Resize the neighbors to rectangular size
  neighbors.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    neighbors[p].resize (max_n_neighbors);
  }

  // Read nearest neighbors lists
  io.read_array (prefix+"neighbors", neighbors);


  // Resize the neighbors to appropriate sizes
  for (long p = 0; p < ncells; p++)
  {
    neighbors[p].resize (n_neighbors[p]);
  }


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

//  io.write_3_vector (prefix+"cells", x, y, z);

//  io.write_3_vector (prefix+"velocities", vx, vy, vz);

    Double2 position_array (position.size(), Double1(3));
    Double2 velocity_array (position.size(), Double1(3));


    for (size_t p=0; p<position.size(); p++)
    {
        position_array[p] = {position[p][0], position[p][1], position[p][2]};
    }

    for (size_t p=0; p<velocity.size(); p++)
    {
        velocity_array[p] = {velocity[p][0], velocity[p][1], velocity[p][2]};
    }

    io.write_array(prefix+"position", position_array);
    io.write_array(prefix+"velocity", velocity_array);

  // Write number of neighbors and neighbors lists

  io.write_list  (prefix+"n_neighbors", n_neighbors);


  //// Resize the neighbors to rectangular size
  //const long max_n_neighbors = *std::max_element (n_neighbors.begin(),
  //                                                n_neighbors.end  () );

  //Long2 neighbors_buffer (ncells, Long1(max_n_neighbors));

  //for (long p = 0; p < ncells; p++)
  //{
  //  for (long n = 0; n < n_neighbors[p]; n++)
  //  {
  //    neighbors_buffer[p][n] = neighbors[p][n];
  //  }
  //}

  io.write_array (prefix+"neighbors", neighbors);


  return (0);

}
