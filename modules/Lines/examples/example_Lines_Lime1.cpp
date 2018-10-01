// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <vector>
using namespace std;
#include <mpi.h>


#include "Lines.hpp"
#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/configure.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/RadiativeTransfer.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"



int main (void)
{

  // Initialize MPI environment

  MPI_Init (NULL, NULL);


  //// Set timer

  //MPI_TIMER timer_TOTAL ("TOTAL");
  //timer_TOTAL.start ();


  const string       cells_file = input_folder + "cells.txt";
  const string n_neighbors_file = input_folder + "n_neighbors.txt";
  const string   neighbors_file = input_folder + "neighbors.txt";
  const string    boundary_file = input_folder + "boundary.txt";
  const string     species_file = input_folder + "species.txt";
  const string   abundance_file = input_folder + "abundance.txt";
  const string temperature_file = input_folder + "temperature.txt";


  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);

  cells.read (cells_file, neighbors_file, boundary_file);

  long nboundary = cells.nboundary;


  LINEDATA linedata;


  SPECIES species (Ncells, Nspec, species_file);
  species.read (abundance_file);


  TEMPERATURE temperature (Ncells);
  temperature.read (temperature_file);


  FREQUENCIES frequencies (Ncells, linedata);
  frequencies.reset (linedata, temperature);

  //frequencies.write("");
  const long nfreq_red = frequencies.nfreq_red;

  
  LEVELS levels (Ncells, linedata);
  
  
  RADIATION radiation (Ncells,
                       Nrays,
                       nfreq_red,
                       nboundary );
  
  radiation.calc_boundary_intensities (cells.bdy_to_cell_nr,
                                       frequencies          );
  
  //radiation.print ("","");
  
  Lines <Dimension, Nrays>(cells,
                           linedata,
                           species,
                           temperature,
                           frequencies,
                           levels,
                           radiation   );
 
  
  // Print results
 
  string tag = "_final";  

  levels.print (tag);
  
  radiation.print (tag);
  
  //// Print total time
  //
  //timer_TOTAL.stop ();
  //timer_TOTAL.print_to_file ();


  // Finalize the MPI environment

  MPI_Finalize ();


  return (0);

}
