// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <limits>
using namespace std;

#include "cells.hpp"
#include "constants.hpp"


///  Constructor for Cells: Allocates memory for cell data
///    @param number_of_cells: number of cells in grid
///    @param number_of_rays: number of rays from each cell
///////////////////////////////////////////////////////////

Cells ::
    Cells (
        const string input_folder)
  : ncells (get_ncells(number_of_cells)),
    nrays  (get_nrays (number_of_rays)),
    rays   (input_folder)
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


  for (long p = 0; p < ncells; p++)
  {
    boundary[p] = false;
    //mirror[p]   = false;

    cell2boundary_nr[p] = ncells;
    boundary2cell_nr[p] = ncells;
  }


}   // END OF CONSTRUCTOR



///  read: read the cells, neighbors and boundary files
///////////////////////////////////////////////////////

int Cells ::
    read (
        const string input_folder)
{

  // Read cell centers and velocities

  ifstream cellsFile (input_folder + "cells.txt");

  for (long p = 0; p < ncells; p++)
  {
    cellsFile >> x[p] >> y[p] >> z[p] >> vx[p] >> vy[p] >> vz[p];
  }

  cellsFile.close();


  // Convert velocities in m/s to fractions for C

  for (long p = 0; p < ncells; p++)
  {
    vx[p] = vx[p] / CC;
    vy[p] = vy[p] / CC;
    vz[p] = vz[p] / CC;
  }


  // Read number of neighbors

  ifstream nNeighborsFile (input_folder + "n_neighbors.txt");

  for (long p = 0; p < ncells; p++)
  {
    nNeighborsFile >> n_neighbors[p];
  }

  nNeighborsFile.close();


  // Resize the neighbors to appropriate sizes

  for (long p = 0; p < ncells; p++)
  {
    neighbors[p].resize (n_neighbors[p]);
  }


  // Read nearest neighbors lists

  ifstream neighborsFile (input_folder + "neighbors.txt");

  for (long p = 0; p < ncells; p++)
  {
    for (long n = 0; n < n_neighbors[p]; n++)
    {
      neighborsFile >> neighbors[p][n];
    }
  }

  neighborsFile.close ();


  // Read boundary list

  ifstream boundaryFile (input_folder + "boundary.txt");

  long index = 0;
  long cell_nr;

  while (boundaryFile >> cell_nr)
  {
    boundary2cell_nr[index]   = cell_nr;
    cell2boundary_nr[cell_nr] = index;

    boundary[cell_nr] = true;

    index++;
  }

  nboundary = index;

  boundaryFile.close();


  return (0);

}




///  setup: setup the cells and their rays
//////////////////////////////////////////

int Cells ::
    setup ()
{

  rays.setup ();


  return (0);

}
