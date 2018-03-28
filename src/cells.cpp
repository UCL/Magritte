// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "declarations.hpp"
#include "cells.hpp"


// Constructor for CELLS: reads input file
// ---------------------------------------

CELLS::CELLS (long number_of_cells)
{

  ncells = number_of_cells;


# if (!FIXED_NCELLS)


    // Standard input

    x = new double[ncells];
    y = new double[ncells];
    z = new double[ncells];

    vx = new double[ncells];
    vy = new double[ncells];
    vz = new double[ncells];

    density = new double[ncells];


    // First derived UV quantities

    UV          = new double[ncells];
    rad_surface = new double[ncells*NRAYS];
    AV          = new double[ncells*NRAYS];


    // Reduction and interpolation

    id      = new long[ncells];
    removed = new bool[ncells];


    // Boundary conditions

    boundary = new bool[ncells];
    mirror   = new bool[ncells];

    intensity = new double[ncells*NRAYS];
    column    = new double[ncells*NRAYS];


    // Geometry

    endpoint = new long[ncells*NRAYS];
    Z        = new double[ncells*NRAYS];

    neighbor    = new long[ncells*NRAYS];
    n_neighbors = new long[ncells];


    // Chemistry

    abundance = new double[ncells*NSPEC];
    rate      = new double[ncells*NREAC];


    // Lines

    pop            = new double[ncells*TOT_NLEV];
    mean_intensity = new double[ncells*TOT_NRAD];


    // Thermal cell data

    temperature_gas      = new double[ncells];
    temperature_dust     = new double[ncells];
    temperature_gas_prev = new double[ncells];

    thermal_ratio      = new double[ncells];
    thermal_ratio_prev = new double[ncells];


# endif


}




// Destructor for CELLS: frees allocated memory
// --------------------------------------------

CELLS::~CELLS ()
{

# if (!FIXED_NCELLS)


    // Standard input

    delete [] x;
    delete [] y;
    delete [] z;

    delete [] vx;
    delete [] vy;
    delete [] vz;

    delete [] density;


    // First derived UV quantities

    delete [] UV;
    delete [] rad_surface;
    delete [] AV;


    // Reduction and interpolation

    delete [] id;
    delete [] removed;


    // Boundary conditions

    delete [] boundary;
    delete [] mirror;

    delete [] intensity;
    delete [] column;


    // Geometry

    delete [] endpoint;
    delete [] Z;

    delete [] neighbor;
    delete [] n_neighbors;


    // Chemistry

    delete [] abundance;
    delete [] rate;


    // Lines

    delete [] pop;
    delete [] mean_intensity;


    // Thermal cell data

    delete [] temperature_gas;
    delete [] temperature_dust;
    delete [] temperature_gas_prev;

    delete [] thermal_ratio;
    delete [] thermal_ratio_prev;


# endif


}
