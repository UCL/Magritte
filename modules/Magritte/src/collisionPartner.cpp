// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <fstream>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "collisionPartner.hpp"
#include "constants.hpp"
#include "species.hpp"
#include "interpolation.hpp"


///  read: read in collision partner data
///    @param[in] io: io object
/////////////////////////////////////////

int CollisionPartner ::
    read (
        const Io &io,
        const int l,
        const int c  )
{

  string prefix = "linedata/lspec_" + to_string (l) + "/";
  prefix = prefix + "colpartner/colpar_" + to_string (c) + "/";


  io.read_number (prefix+".num_col_partner", num_col_partner);
  io.read_word   (prefix+".orth_or_para_H2", orth_or_para_H2);

  io.read_length (prefix+"tmp",  ntmp);
  io.read_length (prefix+"icol", ncol);

  tmp.resize (ntmp);
  io.read_list (prefix+"tmp", tmp);

  icol.resize (ncol);
  jcol.resize (ncol);
  io.read_list (prefix+"icol", icol);
  io.read_list (prefix+"jcol", jcol);

  Ce.resize (ntmp);
  Cd.resize (ntmp);


  for (long t = 0; t < ntmp; t++)
  {
    Ce[t].resize (ncol);
    Cd[t].resize (ncol);

    io.read_array (prefix+"Ce", Ce);
    io.read_array (prefix+"Cd", Cd);
  }


  return (0);

}




///  write: read in collision partner data
///    @param[in] io: io object
//////////////////////////////////////////

int CollisionPartner ::
    write (
        const Io &io,
        const int l,
        const int c  ) const
{

  string prefix = "linedata/lspec_" + to_string (l) + "/";
  prefix = prefix + "colpartner/colpar_" + to_string (c) + "/";


  io.write_number (prefix+".num_col_partner", num_col_partner);
  io.write_word   (prefix+".orth_or_para_H2", orth_or_para_H2);

  io.write_list (prefix+"icol", icol);
  io.write_list (prefix+"jcol", jcol);

  io.write_list (prefix+"tmp", tmp);

  io.write_array (prefix+"Ce", Ce);
  io.write_array (prefix+"Cd", Cd);


  return (0);

}
