// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "boundary.hpp"
#include "Tools/logger.hpp"


const string Boundary::prefix = "Geometry/Boundary/";


///  read: read the input into the data structure
///  @param[in] io: io object
///  @param[in] parameters: model parameters object
///////////////////////////////////////////////////

void Boundary :: read (const Io &io, Parameters &parameters)
{
    cout << "Reading boundary..." << endl;

    ncells = parameters.ncells();

    // Resize boundary
    cell2boundary_nr.resize (ncells);
            boundary.resize (ncells);

    // Initialise
    for (size_t p = 0; p < ncells; p++)
    {
        cell2boundary_nr[p] = ncells;
                boundary[p] = false;
    }

    // Read boundary list
    io.read_length (prefix+"boundary2cell_nr", nboundary);
    boundary2cell_nr.resize (nboundary);
    io.read_list   (prefix+"boundary2cell_nr", boundary2cell_nr);

    // Set model parameter
    parameters.set_nboundary (nboundary);

    // Set helper variables to identify the boundary
    for (size_t b = 0; b < nboundary; b++)
    {
        const long cell_nr = boundary2cell_nr[b];

        cell2boundary_nr[cell_nr] = b;
                boundary[cell_nr] = true;
    }

    // Set boundary conditions
    boundary_condition  .resize (nboundary);
    boundary_temperature.resize (nboundary);

    String1 boundary_condition_str (nboundary);

    io.read_list (prefix+"boundary_temperature", boundary_temperature);
    io.read_list (prefix+"boundary_condition", boundary_condition_str);

    for (size_t b = 0; b < nboundary; b++)
    {
        if (boundary_condition_str[b].compare("zero"))    boundary_condition[b] = Zero;
        if (boundary_condition_str[b].compare("thermal")) boundary_condition[b] = Thermal;
        if (boundary_condition_str[b].compare("cmb"))     boundary_condition[b] = CMB;
    }

}




///  write: write the data structure
///  @param[in] io: io object
////////////////////////////////////////////////

void Boundary :: write (const Io &io) const
{
    cout << "Writing boundary..." << endl;

    io.write_list (prefix+"boundary2cell_nr", boundary2cell_nr);

    String1 boundary_condition_str (nboundary);

    for (size_t b = 0; b < nboundary; b++) switch (boundary_condition[b])
    {
        case Zero    : boundary_condition_str[b] = "zero";
        case Thermal : boundary_condition_str[b] = "thermal";
        case CMB     : boundary_condition_str[b] = "cmb";
    }

    io.write_list (prefix+"boundary_temperature", boundary_temperature);
    io.write_list (prefix+"boundary_condition", boundary_condition_str);

}
