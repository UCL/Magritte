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


	// Set timer

	TIMER timer_TOTAL ("TOTAL");
	timer_TOTAL.start ();


  // Get rank of process and total number of processes

  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


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

	const long nfreq_red = frequencies.nfreq_red;


	LEVELS levels (Ncells, linedata);


  const long START_raypair = ( world_rank   *Nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*Nrays/2)/world_size;

	const long nrays_red = STOP_raypair - START_raypair;

	RADIATION radiation (Ncells, Nrays, nrays_red, nfreq_red,
			                 nboundary, START_raypair);

	// radiation.initialize ();

	radiation.calc_boundary_intensities (cells.bdy_to_cell_nr, frequencies);


	Lines <Dimension, Nrays>
		    (cells, linedata, species, temperature,
				 frequencies, levels, radiation);


	// Print results

	levels.print (output_folder, "");


	// Print total time

	timer_TOTAL.stop ();
	timer_TOTAL.print_to_file ();


  // Finalize the MPI environment

  MPI_Finalize ();


	return (0);

}
