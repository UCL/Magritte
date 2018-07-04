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


  // Get rank of process and total number of processes
	
  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


	const int  dimension = 3;
	const long ncells    = 4*4*4;
	const long nrays     = 12*1*1;
	const long nspec     = 5;

	const string       cells_file = input_folder + "cells.txt";
	const string n_neighbors_file = input_folder + "n_neighbors.txt";
	const string   neighbors_file = input_folder + "neighbors.txt";
	const string    boundary_file = input_folder + "boundary.txt";
	const string     species_file = input_folder + "species.txt";
  const string   abundance_file = input_folder + "abundance.txt";
  const string temperature_file = input_folder + "temperature.txt";


	CELLS <dimension, nrays> cells (ncells, n_neighbors_file);

	cells.read (cells_file, neighbors_file, boundary_file);


	LINEDATA linedata;


	SPECIES species (ncells, nspec, species_file);

	species.read (abundance_file);


	TEMPERATURE temperature (ncells);

	temperature.read (temperature_file);


	FREQUENCIES frequencies (ncells, linedata);

	frequencies.reset (linedata, temperature);

	const long nfreq_red = frequencies.nfreq_red;


	LEVELS levels (ncells, linedata);

	
  const long START_raypair = ( world_rank   *Nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*Nrays/2)/world_size;

	const long nrays_red = STOP_raypair - START_raypair;

	RADIATION radiation (ncells, nrays, nrays, nfreq_red, 0);


	Lines <dimension, nrays>
		    (cells, linedata, species, temperature,
				 frequencies, levels, radiation);


  // Finalize the MPI environment
	
  MPI_Finalize ();	


	return (0);

}
