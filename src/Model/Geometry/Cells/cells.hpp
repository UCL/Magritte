// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


///  CELLS: data structure containing all geometric data
////////////////////////////////////////////////////////

struct Cells
{

public:

    vector<Vector3d> position;
    vector<Vector3d> velocity;

    Long1 n_neighbors;   ///< number of neighbors of each cell
    Long2   neighbors;   ///< cell numbers of the neighbors of each cell


    // Io
    void read  (const Io &io, Parameters &parameters);
    void write (const Io &io                        );


private:

    size_t ncells;                ///< number of cells

    static const string prefix;   ///< prefix to be used in io


};


#endif // __CELLS_HPP_INCLUDED__
