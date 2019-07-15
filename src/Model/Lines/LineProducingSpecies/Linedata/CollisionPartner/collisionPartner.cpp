// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>

#include "collisionPartner.hpp"
#include "Tools/logger.hpp"


const string CollisionPartner::prefix = "Lines/LineProducingSpecies_";


///  read: read in collision partner data
///    @param[in] io: io object
/////////////////////////////////////////

int CollisionPartner ::
    read (
        const Io &io,
        const int l,
        const int c  )
{

  cout << "Reading collisionPartner" << endl;


  const string prefix_lc = prefix + std::to_string (l) + "/Linedata"
                           + "/CollisionPartner_" + std::to_string (c) + "/";


  io.read_number (prefix_lc+".num_col_partner", num_col_partner);
  io.read_word   (prefix_lc+".orth_or_para_H2", orth_or_para_H2);

  io.read_length (prefix_lc+"tmp",  ntmp);
  io.read_length (prefix_lc+"icol", ncol);

  tmp.resize (ntmp);
  io.read_list (prefix_lc+"tmp", tmp);

  icol.resize (ncol);
  jcol.resize (ncol);
  io.read_list (prefix_lc+"icol", icol);
  io.read_list (prefix_lc+"jcol", jcol);

  Ce.resize (ntmp);
  Cd.resize (ntmp);

  for (long t = 0; t < ntmp; t++)
  {
    Ce[t].resize (ncol);
    Cd[t].resize (ncol);

    io.read_array (prefix_lc+"Ce", Ce);
    io.read_array (prefix_lc+"Cd", Cd);
  }


  Ce_intpld.resize (ncol);
  Cd_intpld.resize (ncol);


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

  cout << "Writing collisionPartner" << endl;


  const string prefix_lc = prefix + std::to_string (l) + "/Linedata"
                           + "/CollisionPartner_" + std::to_string (c) + "/";


  io.write_number (prefix_lc+".num_col_partner", num_col_partner);
  io.write_word   (prefix_lc+".orth_or_para_H2", orth_or_para_H2);

  io.write_list (prefix_lc+"icol", icol);
  io.write_list (prefix_lc+"jcol", jcol);

  io.write_list (prefix_lc+"tmp", tmp);

  io.write_array (prefix_lc+"Ce", Ce);
  io.write_array (prefix_lc+"Cd", Cd);


  return (0);

}
