// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "boundary.hpp"
#include "Tools/logger.hpp"


const string Boundary::prefix = "Geometry/Boundary/";


///  read: read the input into the data structure
///  @param[in] io: io object
///  @param[in] parameters: model parameters object
///////////////////////////////////////////////////

void Boundary :: read (const Io &io, Parameters &parameters)
{
  cout << "Reading boundary..." << endl;

  ncells = parameters.ncells();

  // Resize boundary
  cell2boundary_nr.resize (ncells);
          boundary.resize (ncells);

  // Initialise
  for (size_t p = 0; p < ncells; p++)
  {
    cell2boundary_nr[p] = ncells;
            boundary[p] = false;
  }

  // Read boundary list
  io.read_length (prefix+"boundary2cell_nr", nboundary);
  boundary2cell_nr.resize (nboundary);
  io.read_list   (prefix+"boundary2cell_nr", boundary2cell_nr);

  // Set model parameter
  parameters.set_nboundary (nboundary);

  // Set helper variables to identify the boundary
  for (size_t b = 0; b < nboundary; b++)
  {
    const long cell_nr = boundary2cell_nr[b];

    cell2boundary_nr[cell_nr] = b;
            boundary[cell_nr] = true;
  }
}




///  write: write the data structure
///  @param[in] io: io object
////////////////////////////////////////////////

void Boundary :: write (const Io &io) const
{
  cout << "Writing boundary..." << endl;

  io.write_list (prefix+"boundary2cell_nr", boundary2cell_nr);
}
