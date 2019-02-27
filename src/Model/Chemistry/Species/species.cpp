// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "species.hpp"


const string Species::prefix = "Chemistry/Species/";


///  read: read the input into the data structure
///    @paran[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Species ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  io.read_length (prefix+"species",   nspecs);
  io.read_length (prefix+"abundance", ncells);


  parameters.set_ncells (ncells);
  parameters.set_nspecs (nspecs);


   sym.resize (nspecs);
  //mass.resize (nspecs);


  abundance_init.resize (ncells);
       abundance.resize (ncells);


  for (long p = 0; p < ncells; p++)
  {
    abundance_init[p].resize (nspecs);
         abundance[p].resize (nspecs);
  }


  //// Read the species
  //io.read_list (prefix+"species", sym);

  // Read the abundaces of each species in each cell
  io.read_array (prefix+"abundance", abundance);


  // Set initial abundances
  abundance_init = abundance;


  // Get and store species numbers of some inportant species
  //io.read_number (prefix+".nr_e",  nr_e);
  //io.read_number (prefix+".nr_H2", nr_H2);
  //io.read_number (prefix+".nr_HD", nr_HD);


  return (0);

}




///  write: write out the data structure
///  @paran[in] io: io object
/////////////////////////////////////////////////

int Species ::
    write (
        const Io &io) const
{

  Long1 dummy = Long1 (sym.size(), 0);

  io.write_list  (prefix+"species",   dummy);
  io.write_array (prefix+"abundance", abundance);

  io.write_number (prefix+".nr_e", nr_e);
  io.write_number (prefix+".nr_H2", nr_H2);
  io.write_number (prefix+".nr_HD", nr_HD);


  return (0);

}



//
//long Species ::
//    get_species_nr (
//        const string name) const
//{
//
//  // Chech which species corresponds to canonical name
//
//  for (int s = 0; s < nspecs; s++)
//  {
//    if (sym[s] == name)
//    {
//      return s;
//    }
//  }
//
//
//  // Set not found species to be dummy (zeroth species)
//
//  int s = 0;
//
//
//  return s;
//
//}
//
