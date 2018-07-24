// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__

#include <string>

#include "declarations.hpp"


struct CELLS
{

  long ncells;   // number of cells


# if (FIXED_NCELLS)


    // Standard input

    double x[NCELLS],  y[NCELLS],  z[NCELLS];    // coordinates of cell center
    double vx[NCELLS], vy[NCELLS], vz[NCELLS];   // components of velocity field
    double density[NCELLS];                      //  total density in cell


    // First derived UV quantities

    double UV[NCELLS];                  // average UV intensity
    double rad_surface[NCELLS*NRAYS];   // UV radiation inpigning on cell
    double AV[NCELLS*NRAYS];            // Visual extiction


    // Reduction and interpolation

    long id[NCELLS];        // cell nr of associated cell in other grid
    bool removed[NCELLS];   // true when cell is removed


    // Boundary conditions

    bool boundary[NCELLS];   // true if boundary cell
    bool mirror[NCELLS];     // true if reflective boundary

    double intensity[NCELLS*NRAYS];
    double column[NCELLS*NRAYS];


    // Geometry

    long endpoint[NCELLS*NRAYS];   // cell numbers of ray endings
    double Z[NCELLS*NRAYS];        // distance from cell to boundary

    long neighbor[NCELLS*NRAYS];   // cell numbers of neighors
    long n_neighbors[NCELLS];      // number of neighbors


    // Column densities

    double column_H2[NCELLS*NRAYS];    // H2 column density for each ray and cell
    double column_HD[NCELLS*NRAYS];    // HD column density for each ray and cell
    double column_C[NCELLS*NRAYS];     // C  column density for each ray and cell
    double column_CO[NCELLS*NRAYS];    // CO column density for each ray and cell

    double column_tot[NCELLS*NRAYS];   // total column density

    const int spec_size = 1;

    double spectrum[1*NCELLS*NRAYS];   // placeholder for spectrum


    // Chemistry

    double abundance[NCELLS*NSPEC];   // abundance for each species
    double rate[NCELLS*NREAC];        // reaction rate for each reaciton


    // Lines

    double pop[NCELLS*TOT_NLEV];              // level population
    double mean_intensity[NCELLS*TOT_NRAD];   // mean intensity


    // Thermal cell data

    double temperature_gas[NCELLS];
    double temperature_dust[NCELLS];
    double temperature_gas_prev[NCELLS];

    double thermal_ratio[NCELLS];
    double thermal_ratio_prev[NCELLS];


# else


    // Standard input

    double *x,  *y,  *z;    // coordinates of cell center
    double *vx, *vy, *vz;   // components of velocity field
    double *density;        //  total density in cell


    // First derived UV quantities

    double *UV;            // average UV intensity
    double *rad_surface;   // UV radiation inpigning on cell
    double *AV;            // Visual extiction


    // Reduction and interpolation

    long *id;        // cell nr of associated cell in other grid
    bool *removed;   // true when cell is removed


    // Boundary conditions

    bool *boundary;   // true if boundary cell
    bool *mirror;     // true if reflective boundary

    double *intensity;
    double *column;


    // Geometry

    long *endpoint;   // cell numbers of ray endings
    double *Z;        // distance from cell to boundary

    long *neighbor;      // cell numbers of neighors
    long *n_neighbors;   // number of neighbors


    // Column densities

    double *column_H2;    // H2 column density for each ray and cell
    double *column_HD;    // HD column density for each ray and cell
    double *column_C;     // C  column density for each ray and cell
    double *column_CO;    // CO column density for each ray and cell

    double *column_tot;   // total column density

    const int spec_size = 1;
    double *spectrum;   // placeholder for spectrum


    // Chemistry

    double *abundance;   // abundance for each species
    double *rate;        // reaction rate for each reaciton


    // Lines

    double *pop;              // level population
    double *mean_intensity;   // mean intensity


    // Thermal cell data

    double *temperature_gas;
    double *temperature_dust;
    double *temperature_gas_prev;

    double *thermal_ratio;
    double *thermal_ratio_prev;


# endif


  // Constructor: reads grid input file
  // ----------------------------------

  CELLS (long number_of_cells);


  // Destructor: frees allocated memory
  // ----------------------------------

  ~CELLS ();


  // Tools:


  // initialize: initialize cells
  // ----------------------------

  int initialize ();


  // read_input: read input file
  // ---------------------------

  int read_input (std::string inputfile);

  private:
    int read_txt_input(std::string inputfile);
    int read_vtu_input(std::string inputfile);

};


#endif // __CELLS_HPP_INCLUDED__