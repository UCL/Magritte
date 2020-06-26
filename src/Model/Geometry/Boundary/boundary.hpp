// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __BOUNDARY_HPP_INCLUDED__
#define __BOUNDARY_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


enum BoundaryCondition {Zero, Thermal, CMB};


///  Boundary: data structure containing boundary data
//////////////////////////////////////////////////////

struct Boundary
{

  public:

      Long1 boundary2cell_nr;   ///< boundary number of cell
      Long1 cell2boundary_nr;   ///< cell number of boundary

      Bool1 boundary;           ///< true if boundary cell

      vector<BoundaryCondition> boundary_condition;
      Double1                   boundary_temperature;


      // Io
      void read  (const Io &io, Parameters &parameters);
      void write (const Io &io                        ) const;


  private:

      size_t ncells;              ///< number of cells
      size_t nboundary;           ///< number of boundary cells

      static const string prefix;

};


#endif // __BOUNDARY_HPP_INCLUDED__
