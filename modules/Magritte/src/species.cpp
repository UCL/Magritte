// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
using namespace std;

#include "species.hpp"


// Constructor for SPECIES: reads species data file
// ------------------------------------------------

Species ::
    Species (const long number_of_cells,
             const long number_of_species)
 : ncells (number_of_cells),
   nspecs (number_of_species)
{

   sym.resize (nspecs);
  mass.resize (nspecs);

  initial_abundance.resize (nspecs);

  abundance.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    abundance[p].resize (nspecs);
  }


}   // END OF CONSTRUCTOR




///  get_ncells: reads in the number of cells
///  @param[in] inpt_folder: path to folder containing all input data
/////////////////////////////////////////////////////////////////////

static long get_ncells (
    const string input_folder)
{

  long ncells_local;

  ifstream ncellsFile (input_folder + "ncells.txt");

  ncellsFile >> ncells_local;

  cellsFile.close();


  return ncells_local;

}




///  get_nspecs: reads in the number of cells
///  @param[in] inpt_folder: path to folder containing all input data
/////////////////////////////////////////////////////////////////////

static long get_nspecs (
    const string input_folder)
{

  long nspecs_local;

  ifstream nspecsFile (input_folder + "nspecs.txt");

  nspecsFile >> nspecs_local;

  specsFile.close();


  return nspecs_local;

}




int Species ::
    read (
        const string input_folder)
{

  // Read the species

  // First species is a dummy for when a species is not found
  sym[0]               = "dummy0";
  initial_abundance[0] = 0.0;

  ifstream speciesFile (input_folder + "species.txt");

  for (long s = 1; s < nspecs-1; s++)
  {
    speciesFile >> sym[s] >> mass[s] >> initial_abundance[s];
  }

  speciesFile.close ();

  // Last species is a dummy with abundance 1.0 everywhere
  sym[nspecs-1]               = "dummy1";
  initial_abundance[nspecs-1] = 1.0;


  // Read the abundaces of each species in each cell

  ifstream abundanceFile (input_folder + "abundance.txt");

  for (long p = 0; p < ncells; p++)
  {
    for (long s = 0; s < nspecs; s++)
    {
      abundanceFile >> abundance[p][s];
    }
  }

  abundanceFile.close ();


  return (0);

}




int Species ::
    setup ()
{

  // Get and store species numbers of some inportant species

  nr_e    = get_species_nr ("e-");     // species nr for electrons
  nr_H2   = get_species_nr ("H2");     // species nr for H2
  nr_HD   = get_species_nr ("HD");     // species nr for HD
  nr_C    = get_species_nr ("C");      // species nr for C
  nr_H    = get_species_nr ("H");      // species nr for H
  nr_H2x  = get_species_nr ("H2+");    // species nr for H2+
  nr_HCOx = get_species_nr ("HCO+");   // species nr for HCO+
  nr_H3x  = get_species_nr ("H3+");    // species nr for H3+
  nr_H3Ox = get_species_nr ("H3O+");   // species nr for H3O+
  nr_Hex  = get_species_nr ("He+");    // species nr for He+
  nr_CO   = get_species_nr ("CO");     // species nr for CO


  return (0);
}




///  get_species_nr: get number corresponding to given species symbol
/////////////////////////////////////////////////////////////////////

int Species ::
    get_species_nr (
        const string name)
{

  // Chech which species corresponds to name

  for (int s = 0; s < nspecs; s++)
  {
    if (sym[s] == name)
    {
      return s;
    }
  }


  return (-1);

}
