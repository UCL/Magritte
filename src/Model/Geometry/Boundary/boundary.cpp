// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "boundary.hpp"
#include "Tools/logger.hpp"


const string Boundary::prefix = "Geometry/Boundary/";


///  read: read the input into the data structure
///  @paran[in] io: io object
///  @paran[in] parameters: model parameters object
///////////////////////////////////////////////////

int Boundary ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  write_to_log ("Reading boundary");


  ncells = parameters.ncells();


  // Resize boundary
  boundary2cell_nr.resize (ncells);
  cell2boundary_nr.resize (ncells);
          boundary.resize (ncells);


  // Initialise
  for (long p = 0; p < ncells; p++)
  {
    cell2boundary_nr[p] = ncells;
    boundary2cell_nr[p] = ncells;
            boundary[p] = false;
  }


  // Read boundary list
  io.read_length (prefix+"boundary2cell_nr", nboundary);
  io.read_list   (prefix+"boundary2cell_nr", boundary2cell_nr);


  // Set model parameter
  parameters.set_nboundary (nboundary);


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

int Boundary ::
    write (
        const Io &io) const
{

  write_to_log ("Writing boundary");


  io.write_list (prefix+"boundary2cell_nr", boundary2cell_nr);


  return (0);

}
