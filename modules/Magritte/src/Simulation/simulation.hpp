// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SIMULATION_HPP_INCLUDED__
#define __SIMULATION_HPP_INCLUDED__


#include "types.hpp"
#include "io.hpp"
#include "model.hpp"


///  Model: a distributed data structure for Magritte's model data
//////////////////////////////////////////////////////////////////

struct Simulation
{

    Model model;

    // Constructor
    Simulation (
        Model &model );


    int compute_radiation_field () const;

    int compute_level_populations ();


    // Io
    int read (
        const Io &io);

    int write (
        const Io &io) const;


};


#endif // __SIMULATION_HPP_INCLUDED__
