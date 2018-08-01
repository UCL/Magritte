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
#include "../src/configure.hpp"
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

	//const int  Dimension = 3;
	//const long ncells    = 5;
	//const long Nrays     = 12;
	//const long nspec     = 5;


	//const string project_folder = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data";

	//const string       cells_file = project_folder + "cells.txt";
	//const string n_neighbors_file = project_folder + "n_neighbors.txt";
	//const string   neighbors_file = project_folder + "neighbors.txt";
	//const string    boundary_file = project_folder + "boundary.txt";
	//const string     species_file = project_folder + "species.txt";
  //const string   abundance_file = project_folder + "abundance.txt";
  //const string temperature_file = project_folder + "temperature.txt";

	const string       cells_file = input_folder + "cells.txt";
	const string n_neighbors_file = input_folder + "n_neighbors.txt";
	const string   neighbors_file = input_folder + "neighbors.txt";
	const string    boundary_file = input_folder + "boundary.txt";
	const string     species_file = input_folder + "species.txt";
  const string   abundance_file = input_folder + "abundance.txt";
  const string temperature_file = input_folder + "temperature.txt";

	CELLS<Dimension, Nrays> cells (Ncells, n_neighbors_file);

	cells.read (cells_file, neighbors_file, boundary_file);

	long nboundary = cells.nboundary;


	LINEDATA linedata;   // object containing line data


	TEMPERATURE temperature (Ncells);

	temperature.read (temperature_file);


  FREQUENCIES frequencies (Ncells, linedata);

	frequencies.reset (linedata, temperature);

	const long nfreq_red = frequencies.nfreq_red;


	RADIATION radiation (Ncells, Nrays, nfreq_red, nboundary);


	LINES lines (Ncells, linedata);


	const long nfreq_scat = 1;
	SCATTERING scattering (Nrays, nfreq_scat, nfreq_red);


	MPI_TIMER timer_RT ("RT");
	timer_RT.start ();

	RadiativeTransfer <Dimension,Nrays>
		                (cells, temperature, frequencies,
										 lines, scattering, radiation);

	timer_RT.stop ();
	timer_RT.print ();


  MPI_Finalize ();


  return (0);

}
