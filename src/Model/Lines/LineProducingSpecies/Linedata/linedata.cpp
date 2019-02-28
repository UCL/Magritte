// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
using std::string;
#include <Eigen/Core>

#include "linedata.hpp"
#include "Tools/constants.hpp"
#include "Functions/interpolation.hpp"


const string Linedata::prefix = "Lines/LineProducingSpecies_";


///  read: read in line data
///    @param[in] io: io object
///    @param[in] l: nr of line producing species
/////////////////////////////////////////////////

int Linedata ::
    read (
        const Io &io,
        const int l  )
{

  const string prefix_l = prefix + std::to_string (l) + "/Linedata/";


  io.read_number (prefix_l+".num", num);
  io.read_word   (prefix_l+".sym", sym);

  io.read_length (prefix_l+"energy",    nlev);
  io.read_length (prefix_l+"frequency", nrad);

  irad.resize (nrad);
  jrad.resize (nrad);

  io.read_list (prefix_l+"irad", irad);
  io.read_list (prefix_l+"jrad", jrad);

  energy.resize (nlev);
  weight.resize (nlev);

  io.read_list (prefix_l+"energy", energy);
  io.read_list (prefix_l+"weight", weight);

  frequency.resize (nrad);

  io.read_list (prefix_l+"frequency", frequency);

  A.resize  (nrad);
  Bs.resize (nrad);
  Ba.resize (nrad);

  io.read_list (prefix_l+"A",  A);
  io.read_list (prefix_l+"Bs", Bs);
  io.read_list (prefix_l+"Ba", Ba);


  // Get ncolpar
  io.read_length (prefix_l+"CollisionPartner_", ncolpar);


  colpar.resize (ncolpar);

  for (int c = 0; c < ncolpar; c++)
  {
    colpar[c].read (io, l, c);
  }


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
///    @param[in] l: nr of line producing species
/////////////////////////////////////////////////

int Linedata ::
    write (
        const Io &io,
        const int l  ) const
{

  const string prefix_l = prefix + std::to_string (l) + "/Linedata/";


  io.write_number (prefix_l+".num", num);
  io.write_word   (prefix_l+".sym", sym);

  io.write_list (prefix_l+"irad", irad);
  io.write_list (prefix_l+"jrad", jrad);

  io.write_list (prefix_l+"energy", energy);
  io.write_list (prefix_l+"weight", weight);

  io.write_list (prefix_l+"frequency", frequency);

  io.write_list (prefix_l+"A",  A);
  io.write_list (prefix_l+"Bs", Bs);
  io.write_list (prefix_l+"Ba", Ba);


  for (int c = 0; c < ncolpar; c++)
  {
    colpar[c].write (io, l, c);
  }


  return (0);

}
