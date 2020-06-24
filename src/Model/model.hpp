// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MODEL_HPP_INCLUDED__
#define __MODEL_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Model/parameters.hpp"
#include "Model/Geometry/geometry.hpp"
#include "Model/Chemistry/chemistry.hpp"
#include "Model/Lines/lines.hpp"
#include "Model/Thermodynamics/thermodynamics.hpp"
#include "Model/Radiation/radiation.hpp"


///  Model: a distributed data structure for Magritte's model data
//////////////////////////////////////////////////////////////////

struct Model
{
    // Log file writer
    Logger logger;

    // Bookkeeping
    Parameters parameters;

    // Science
    Geometry       geometry;
    Chemistry      chemistry;
    Lines          lines;
    Thermodynamics thermodynamics;
    Radiation      radiation;

    // Constructor
    Model();

    // Io
    void read  (const Io &io);
    void write (const Io &io);
};


#endif // __MODEL_HPP_INCLUDED__
