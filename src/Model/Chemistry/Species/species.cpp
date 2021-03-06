// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "species.hpp"
#include "Tools/logger.hpp"


const string Species::prefix = "Chemistry/Species/";


///  read: read the input into the data structure
///    @param[in] io: io object
///    @param[in] parameters: model parameters object
/////////////////////////////////////////////////////

void Species :: read (const Io &io, Parameters &parameters)
{
    cout << "Reading species..." << endl;

    io.read_length (prefix+"species",   nspecs);
    io.read_length (prefix+"abundance", ncells);

    parameters.set_ncells (ncells);
    parameters.set_nspecs (nspecs);

     sym.resize (nspecs);
    //mass.resize (nspecs);

    abundance_init.resize (ncells);
         abundance.resize (ncells);

    for (size_t p = 0; p < ncells; p++)
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
}




///  write: write out the data structure
///  @param[in] io: io object
/////////////////////////////////////////////////

void Species :: write (const Io &io) const
{
    cout << "Writing species..." << endl;

    Long1 dummy = Long1 (sym.size(), 0);

    io.write_list  (prefix+"species",   dummy    );
    io.write_array (prefix+"abundance", abundance);

    //io.write_number (prefix+".nr_e",  nr_e );
    //io.write_number (prefix+".nr_H2", nr_H2);
    //io.write_number (prefix+".nr_HD", nr_HD);
}
