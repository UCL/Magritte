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
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/RadiativeTransfer.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"



int main (void)
{

	const int  Dimension = 3;
	const long ncells    = 4*4*4;
	const long Nrays     = 12*1*1;
	const long nspec     = 5;

	const string project_folder = "/home/frederik/Dropbox/Astro/MagritteProjects/test1/";

	const string       cells_file = project_folder + "cells.txt";
	const string n_neighbors_file = project_folder + "n_neighbors.txt";
	const string   neighbors_file = project_folder + "neighbors.txt";
	const string    boundary_file = project_folder + "boundary.txt";
	const string     species_file = project_folder + "species.txt";
  const string   abundance_file = project_folder + "abundance.txt";
  const string temperature_file = project_folder + "temperature.txt";


	CELLS<Dimension, Nrays> cells (ncells, n_neighbors_file);

	cells.read (cells_file, neighbors_file, boundary_file);


	LINEDATA linedata;


	SPECIES species (ncells, nspec, species_file);

	species.read (abundance_file);


	TEMPERATURE temperature (ncells);

	temperature.read (temperature_file);


	FREQUENCIES frequencies (ncells, linedata);

	frequencies.reset (linedata, temperature);

	const long nfreq = frequencies.nfreq;


	LEVELS levels (ncells, linedata);


	RADIATION radiation (ncells, Nrays, nfreq);
	

	Lines<Dimension, Nrays> (cells, linedata, species, temperature, frequencies, levels, radiation);


	return (0);

}
