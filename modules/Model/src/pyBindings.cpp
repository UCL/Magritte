// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "model.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
namespace py = pybind11;


PYBIND11_MODULE (magritte, magritte_module)
{
  // Module docstring
  magritte_module.doc() = "magritte module";


  // Binding to linedata struct
  py::class_<Model> (magritte_module, "Model")
      // constructor
      .def (py::init<long>());
      // attributes
      // .def_readonly  ("nlev"      , &LINEDATA::nlev)
      // .def_readonly  ("nrad"      , &LINEDATA::nrad)
      // .def_readwrite ("irad"      , &LINEDATA::irad)
      // .def_readwrite ("jrad"      , &LINEDATA::jrad)
      // .def_readwrite ("energy"    , &LINEDATA::energy)
      // .def_readwrite ("weight"    , &LINEDATA::weight)
      // .def_readwrite ("population", &LINEDATA::population)
      // .def_readwrite ("frequency" , &LINEDATA::frequency)
      // .def_readwrite ("emissivity", &LINEDATA::emissivity)
      // .def_readwrite ("opacity"   , &LINEDATA::opacity)
      // .def_readwrite ("A"         , &LINEDATA::A)
      // .def_readwrite ("B"         , &LINEDATA::B)
      // // functions
      // .def ("compute_populations_LTE",        &LINEDATA::compute_populations_LTE)
      // .def ("compute_emissivity_and_opacity", &LINEDATA::compute_emissivity_and_opacity);
}
