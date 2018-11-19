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


  CELLS <DIMENSION, NRAYS> cells (
      NCELLS,
      n_neighbors_file           );

  cells.read         (
      cells_file,
      neighbors_file,
      boundary_file  );


  LINEDATA linedata (input_folder + "linedata/");


  SPECIES species (
      NCELLS,
      NSPEC,
      species_file);

  species.read (abundance_file);


  TEMPERATURE temperature (NCELLS);
  temperature.read (temperature_file);


  FREQUENCIES frequencies (NCELLS, linedata);
  frequencies.reset (linedata, temperature);

  
  LEVELS levels (
      NCELLS,
      linedata  );
  
  
  RADIATION radiation (
      NCELLS,
      NRAYS,
      frequencies.nfreq_red,
      cells.nboundary );
  

  radiation.calc_boundary_intensities (
      cells.boundary2cell_nr,
      cells.cell2boundary_nr,
      frequencies                     );
  

  levels.compute_all <DIMENSION, NRAYS>(
      cells,
      linedata,
      species,
      temperature,
      frequencies,
      radiation                        );
  
  
  // Print results
 
  string tag = "_final";  

  levels.print (tag);
  
  radiation.print (tag);

  frequencies.print (tag);


  //// Print total time
  //
  //timer_TOTAL.stop ();
  //timer_TOTAL.print_to_file ();

  //cout << linedata.A[0] << endl;


  // Finalize the MPI environment

  MPI_Finalize ();


  return (0);

}
