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

void Cells :: read (const Io &io, Parameters &parameters)
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

    for (size_t p = 0; p < position.size(); p++)
    {
        position[p] = {position_array[p][0], position_array[p][1], position_array[p][2]};
    }

    for (size_t p = 0; p < velocity.size(); p++)
    {
        velocity[p] = {velocity_array[p][0], velocity_array[p][1], velocity_array[p][2]};
    }


    // Read number of neighbors
    n_neighbors.resize (ncells);

    io.read_list (prefix+"n_neighbors", n_neighbors);


    // Resize the neighbors to rectangular size
//    const long max_n_neighbors = *std::max_element (n_neighbors.begin(),
//                                                    n_neighbors.end  () );

    size_t tot_n_neighbors = 0;
    for (size_t p = 0; p < ncells; p++)
    {
        tot_n_neighbors += n_neighbors[p];
    }

    neighbors.resize (ncells);

    Long1 lin_neighbors;
    lin_neighbors.reserve (tot_n_neighbors);

//    for (size_t p = 0; p < ncells; p++)
//    {
//        neighbors[p].resize (max_n_neighbors);
//    }

    // Read nearest neighbors lists
//    io.read_array (prefix+"neighbors", neighbors);
    io.read_list (prefix+"neighbors", lin_neighbors);


    // Resize the neighbors to appropriate sizes
    Long1::iterator index = lin_neighbors.begin();
    for (size_t p = 0; p < ncells; p++)
    {
        neighbors[p].reserve (n_neighbors[p]);
        neighbors[p].insert  (neighbors[p].begin(), index, index+n_neighbors[p]);
        index += n_neighbors[p];
    }
}




///  Writer for the Cells data to the Io object
///  @param[in] io : io object
///////////////////////////////////////////////

void Cells :: write (const Io &io)
{
    cout << "Writing cells..." << endl;

    // Write cell positions and velocities
    Double2 position_array (position.size(), Double1(3));
    Double2 velocity_array (position.size(), Double1(3));

    for (size_t p = 0; p < position.size(); p++)
    {
        position_array[p] = {position[p][0], position[p][1], position[p][2]};
    }

    for (size_t p = 0; p < velocity.size(); p++)
    {
        velocity_array[p] = {velocity[p][0], velocity[p][1], velocity[p][2]};
    }

    io.write_array(prefix+"position", position_array);
    io.write_array(prefix+"velocity", velocity_array);

    size_t tot_n_neighbors = 0;

    // Might not be initialized at this point, hence the resize!
    n_neighbors.resize (neighbors.size());

    // Make sure n_neighbours is properly set
    for (size_t p = 0; p < neighbors.size(); p++)
    {
            n_neighbors[p]  =   neighbors[p].size();
        tot_n_neighbors    += n_neighbors[p];
    }

    // Write number of neighbors and neighbors lists
    io.write_list  (prefix+"n_neighbors", n_neighbors);

    Long1 lin_neighbors;
    lin_neighbors.reserve (tot_n_neighbors);

      // Resize the neighbors to rectangular size
//    const long max_n_neighbors = *std::max_element (n_neighbors.begin(),
//                                                    n_neighbors.end  () );

//    cout << "max_n_neighbours = " << max_n_neighbors << endl;

    for (size_t p = 0; p < neighbors.size(); p++)
    {
//        neighbors[p].resize (max_n_neighbors);
        lin_neighbors.insert(lin_neighbors.end(), neighbors[p].begin(), neighbors[p].end());
    }


      io.write_list (prefix+"neighbors", lin_neighbors);
//    io.write_array (prefix+"neighbors", neighbors);

    // Resize the neighbors to appropriate sizes
//    for (size_t p = 0; p < neighbors.size(); p++)
//    {
//        neighbors[p].resize (n_neighbors[p]);
//    }
}
