// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "linedata.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
namespace py = pybind11;


PYBIND11_MODULE(linedata, linedata_module)
{
  // Module docstring
  linedata_module.doc() = "linedata module";


  // Binding to linedata struct
  py::class_<LINEDATA>(linedata_module, "LineData")
      // constructor
      .def (py::init<int, int>())
      // attributes
      .def_readonly  ("nlev"      , &LINEDATA::nlev)
      .def_readonly  ("nrad"      , &LINEDATA::nrad)
      .def_readwrite ("irad"      , &LINEDATA::irad)
      .def_readwrite ("jrad"      , &LINEDATA::jrad)
      .def_readwrite ("energy"    , &LINEDATA::energy)
      .def_readwrite ("weight"    , &LINEDATA::weight)
      .def_readwrite ("population", &LINEDATA::population)
      .def_readwrite ("frequency" , &LINEDATA::frequency)
      .def_readwrite ("emissivity", &LINEDATA::emissivity)
      .def_readwrite ("opacity"   , &LINEDATA::opacity)
      .def_readwrite ("A"         , &LINEDATA::A)
      .def_readwrite ("B"         , &LINEDATA::B)
      // functions
      .def ("compute_populations_LTE",        &LINEDATA::compute_populations_LTE)
      .def ("compute_emissivity_and_opacity", &LINEDATA::compute_emissivity_and_opacity);
}
