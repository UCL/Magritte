// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "quadrature.hpp"


const string Quadrature::prefix = "Lines/Quadrature/";


///  read: read in data structure
///    @param[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Quadrature ::
    read (
        const Io         &io,
              Parameters &parameters)
{


  io.read_length (prefix+"weights", nquads);


  parameters.set_nquads (nquads);


  weights.resize (nquads);
  roots.resize   (nquads);

  io.read_list (prefix+"weights", weights);
  io.read_list (prefix+"roots",   roots  );


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
////////////////////////////////////

int Quadrature ::
    write (
        const Io &io) const
{

  io.write_list (prefix+"weights", weights);
  io.write_list (prefix+"roots",   roots  );


  return (0);

}
