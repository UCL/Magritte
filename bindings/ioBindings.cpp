// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Io/io.hpp"
#include "Io/io_text.hpp"
#include "Io/io_Python.hpp"
#include "Tools/types.hpp"
#include "Model/model.hpp"
#include "Simulation/simulation.hpp"

#include "pybind11/pybind11.h"
namespace py = pybind11;


PYBIND11_MODULE (ioMagritte, module)
{

  // Module docstring
  module.doc() = "Io module";


  // Io
  py::class_<Io> (module, "Io");


  // IoText
  py::class_<IoText, Io> (module, "IoText")
      // attributes
      .def_readonly ("io_file", &IoText::io_file)
      // constructor
      .def (py::init<const string &>());


  // IoPython
  py::class_<IoPython, Io> (module, "IoPython")
      // attributes
      .def_readonly ("implementation", &IoPython::implementation)
      .def_readonly ("io_file",        &IoPython::io_file)
      // constructor
      .def (py::init<const string &, const string &>());


}