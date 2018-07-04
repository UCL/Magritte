// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
#include <string>
#include <mpi.h>
#include <omp.h>

#include "../src/RadiativeTransfer.hpp"
#include "../src/types.hpp"
#include "../src/GridTypes.hpp"
#include "../src/timer.hpp"
#include "../src/cells.hpp"
#include "../src/temperature.hpp"
#include "../src/frequencies.hpp"
#include "../src/radiation.hpp"
#include "../src/lines.hpp"
#include "../../Lines/src/linedata.hpp"


int main (void)
{


	MPI_Init (NULL, NULL);

	const int  Dimension = 3;
	const long ncells    = 5;
	const long Nrays     = 12;
	const long nspec     = 5;


	const string project_folder = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data";

	const string       cells_file = project_folder + "cells.txt";
	const string n_neighbors_file = project_folder + "n_neighbors.txt";
	const string   neighbors_file = project_folder + "neighbors.txt";
	const string    boundary_file = project_folder + "boundary.txt";
	const string     species_file = project_folder + "species.txt";
  const string   abundance_file = project_folder + "abundance.txt";
  const string temperature_file = project_folder + "temperature.txt";


  // long nrays = Nrays/(2*world_size);

	CELLS<Dimension, Nrays> cells (ncells, n_neighbors_file);

	cells.read (cells_file, neighbors_file, boundary_file);


	LINEDATA linedata;   // object containing line data



	TEMPERATURE temperature (ncells);

	temperature.read (temperature_file);


  FREQUENCIES frequencies (ncells, linedata);

	frequencies.reset (linedata, temperature);

	const long nfreq_red = frequencies.nfreq_red;


  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

  const long START_raypair = ( world_rank   *Nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*Nrays/2)/world_size;

	const long nrays_red = STOP_raypair - START_raypair;

	RADIATION radiation (ncells, Nrays, nrays_red, nfreq_red, START_raypair);
	

	LINES lines (ncells, linedata);


	const long nfreq_scat = 1;
	SCATTERING scattering (Nrays, nfreq_scat, nfreq_red);


	RadiativeTransfer <Dimension,Nrays>
		                (cells, temperature, frequencies,
										 lines, scattering, radiation);



  MPI_Finalize ();	


  return(0);

}
