// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MAGRITTE_TYPES_HPP_INCLUDED__
#define __MAGRITTE_TYPES_HPP_INCLUDED__


// #include "Eigen/Dense"
#include <omp.h>

// Include CELLS class
#include "cells.hpp"
// Include RAYS class
#include "rays.hpp"
// Include SPECIES class
#include "species.hpp"
// Include REACTIONS class
#include "reactions.hpp"
// Include LINES class
#include "lines.hpp"


struct COLUMN_DENSITIES
{

# if (FIXED_NCELLS)

    double H2[NCELLS*NRAYS];    // H2 column density
    double HD[NCELLS*NRAYS];    // HD column density
    double C[NCELLS*NRAYS];     // C  column density
    double CO[NCELLS*NRAYS];    // CO column density

    double tot[NCELLS*NRAYS];   // total column density

# else

    double *H2;    // H2 column density
    double *HD;    // HD column density
    double *C;     // C  column density
    double *CO;    // CO column density

    double *tot;   // total column density

# endif


    COLUMN_DENSITIES (long number_of_cells)
    {

#     if (!FIXED_NCELLS)

        H2  = new double[number_of_cells*NRAYS];   // H2 column density for each ray and cell
        HD  = new double[number_of_cells*NRAYS];   // HD column density for each ray and cell
        C   = new double[number_of_cells*NRAYS];   // C  column density for each ray and cell
        CO  = new double[number_of_cells*NRAYS];   // CO column density for each ray and cell

        tot = new double[number_of_cells*NRAYS];   // CO column density for each ray and cell

#     endif

    }

    ~COLUMN_DENSITIES ()
    {

#     if (!FIXED_NCELLS)

        delete [] H2;
        delete [] HD;
        delete [] C ;
        delete [] CO;

        delete [] tot;

#     endif

    }



};




struct TIMER
{

  double duration;


  void initialize()
  {
    duration = 0.0;
  }


  void start()
  {
    duration -= omp_get_wtime();
  }


  void stop()
  {
    duration += omp_get_wtime();
  }

};




struct TIMERS
{

  TIMER total;
  TIMER chemistry;
  TIMER level_pop;


  void initialize()
  {
    total.initialize();
    chemistry.initialize();
    level_pop.initialize();
  }

};



struct NITERATIONS
{

  int tb;
  int rt;

  void initialize()
  {
    tb = 0;
    rt = 0;
  }

};

// Include LINES class
// #include "lines.hpp"



#endif //__MAGRITTE_TYPES_HPP_INCLUDED__
