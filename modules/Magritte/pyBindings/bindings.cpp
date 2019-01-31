// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "io.hpp"
#include "io_text.hpp"
#include "model.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
namespace py = pybind11;


PYBIND11_MODULE (pyMagritte, module)
{
  // Module docstring
  module.doc() = "Magritte module";

  // Io
  py::class_<Io> (module, "Io");

  // IoText
  py::class_<IoText, Io> (module, "IoText")
      // attributes
      .def_readonly ("io_file", &IoText::io_file)
      // constructor
      .def (py::init<const string &>());
      // functions
      //.def ("get_number",      &IoText::get_number)
      //.def ("get_length",      &IoText::get_length)
      //.def ("read_list",       &IoText::read_list)
      //.def ("read_3_vector",   &IoText::read_3_vector)
      //.def ("write_3_vector",  &IoText::write_3_vector);
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
      // .def ("compute_populations_LTE",        Y&LINEDATA::compute_populations_LTE)
      // .def ("compute_emissivity_and_opacity", &LINEDATA::compute_emissivity_and_opacity);

  // Model
  py::class_<Model> (module, "Model")
      // attributes
      .def_readwrite ("cells", &Model::cells)
      // constructor
      .def (py::init<const Io &>());

  // Cells
  py::class_<Cells> (module, "Cells")
      // attributes
      .def_readwrite ("ncells",           &Cells::ncells)
      .def_readwrite ("nboundary",        &Cells::nboundary)
      .def_readwrite ("rays",             &Cells::rays)
      .def_readwrite ("x",                &Cells::x)
      .def_readwrite ("y",                &Cells::y)
      .def_readwrite ("z",                &Cells::z)
      .def_readwrite ("vx",               &Cells::vx)
      .def_readwrite ("vy",               &Cells::vy)
      .def_readwrite ("vz",               &Cells::vz)
      .def_readwrite ("boundary2cell_nr", &Cells::boundary2cell_nr)
      // constructor
      .def (py::init<const Io &>())
      // functions
      .def ("write",                      &Cells::write);

  // Rays
  py::class_<Rays> (module, "Rays")
      // attributes
      .def_readwrite ("nrays", &Rays::nrays)
      .def_readwrite ("x",     &Rays::x)
      .def_readwrite ("y",     &Rays::y)
      .def_readwrite ("z",     &Rays::z)
      // constructor
      .def (py::init<const Io &>())
      // functions
      .def ("write",           &Rays::write);

}
