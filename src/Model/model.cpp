// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "model.hpp"
#include "Io/io.hpp"
#include "Tools/logger.hpp"


///  Reader for the Model data
///    @param[in] io : io object to read with
/////////////////////////////////////////////

int Model ::
   read (
      const Io &io)
{

  cout << "                                           " << endl;
  cout << "-------------------------------------------" << endl;
  cout << "  Reading Model...                         " << endl;
  cout << "-------------------------------------------" << endl;
  cout << "                                           " << endl;


  geometry.read       (io, parameters);

  thermodynamics.read (io, parameters);

  chemistry.read      (io, parameters);

  lines.read          (io, parameters);

  radiation.read      (io, parameters);


  cout << "                                           " << endl;
  cout << "-------------------------------------------" << endl;
  cout << "  Model parameters:                        " << endl;
  cout << "-----------------------------------------  " << endl;
  cout << "  ncells     = " << parameters.ncells     () << endl;
  cout << "  ncameras   = " << parameters.ncameras   () << endl;
  cout << "  nrays      = " << parameters.nrays      () << endl;
  cout << "  nrays_red  = " << parameters.nrays_red  () << endl;
  cout << "  nboundary  = " << parameters.nboundary  () << endl;
  cout << "  nfreqs     = " << parameters.nfreqs     () << endl;
  cout << "  nfreqs_red = " << parameters.nfreqs_red () << endl;
  cout << "  nspecs     = " << parameters.nspecs     () << endl;
  cout << "  nlspecs    = " << parameters.nlspecs    () << endl;
  cout << "  nlines     = " << parameters.nlines     () << endl;
  cout << "  nquads     = " << parameters.nquads     () << endl;
  cout << "-------------------------------------------" << endl;
  cout << "                                           " << endl;


  return (0);

}




///  Writer for the Model data
///    @param[in] io : io object to write with
//////////////////////////////////////////////

int Model ::
   write (
      const Io &io) const
{

  cout << "Writing Model" << endl;

  parameters.write     (io);

  geometry.write       (io);

  thermodynamics.write (io);

  chemistry.write      (io);

  lines.write          (io);

  radiation.write      (io);


 return (0);

}
