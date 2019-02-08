// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "io.hpp"
#include "io_text.hpp"
#include "io_Python.hpp"
#include "types.hpp"
#include "model.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"
//#include "pybind11/eigen.h"
//#include "pybind11/stl.h"
namespace py = pybind11;


PYBIND11_MODULE (pyMagritte, module)
{

  py::bind_vector<Long1>   (module, "Long1");
  py::bind_vector<Long2>   (module, "Long2");
  py::bind_vector<Long3>   (module, "Long3");

  py::bind_vector<Double1> (module, "Double1");
  py::bind_vector<Double2> (module, "Double2");
  py::bind_vector<Double3> (module, "Double3");

  py::bind_vector<String1> (module, "String1");

  py::bind_vector<vector<Linedata>>         (module, "vLinedata");
  py::bind_vector<vector<CollisionPartner>> (module, "vCollisionPartner");

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

  // IoPython
  py::class_<IoPython, Io> (module, "IoPython")
      // attributes
      .def_readonly ("implementation", &IoPython::implementation)
      .def_readonly ("io_file",        &IoPython::io_file)
      // constructor
      .def (py::init<const string &, const string &>());

  // Model
  py::class_<Model> (module, "Model")
      // attributes
      .def_readwrite ("nlspecs",     &Model::nlspecs)
      .def_readwrite ("cells",       &Model::cells)
      .def_readwrite ("temperature", &Model::temperature)
      .def_readwrite ("species",     &Model::species)
      .def_readwrite ("linedata",    &Model::linedata)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                  &Model::read)
      .def ("write",                 &Model::write);

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
      .def_readwrite ("n_neighbors",      &Cells::n_neighbors)
      .def_readwrite ("neighbors",        &Cells::neighbors)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                       &Cells::read)
      .def ("write",                      &Cells::write);

  // Rays
  py::class_<Rays> (module, "Rays")
      // attributes
      .def_readwrite ("nrays", &Rays::nrays)
      .def_readwrite ("x",     &Rays::x)
      .def_readwrite ("y",     &Rays::y)
      .def_readwrite ("z",     &Rays::z)
      // constructor
      .def (py::init())
      // functions
      .def ("read",            &Rays::read)
      .def ("write",           &Rays::write);

  // Temperature
  py::class_<Temperature> (module, "Temperature")
      // attributes
      .def_readwrite ("ncells", &Temperature::ncells)
      .def_readwrite ("gas",    &Temperature::gas)
      .def_readwrite ("dust",   &Temperature::dust)
      .def_readwrite ("vturb2", &Temperature::vturb2)
      // constructor
      .def (py::init())
      // functions
      .def ("read",             &Temperature::read)
      .def ("write",            &Temperature::write);

  // Species
  py::class_<Species> (module, "Species")
      // attributes
      .def_readwrite ("ncells",    &Species::ncells)
      .def_readwrite ("nspecs",    &Species::nspecs)
      .def_readwrite ("abundance", &Species::abundance)
      .def_readwrite ("sym",       &Species::sym)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                &Species::read)
      .def ("write",               &Species::write);

  // Linedata
  py::class_<Linedata> (module, "Linedata")
      // attributes
      .def_readwrite ("num",       &Linedata::num)
      .def_readwrite ("sym",       &Linedata::sym)
      .def_readwrite ("nlev",      &Linedata::nlev)
      .def_readwrite ("nrad",      &Linedata::nrad)
      .def_readwrite ("irad",      &Linedata::irad)
      .def_readwrite ("jrad",      &Linedata::jrad)
      .def_readwrite ("energy",    &Linedata::energy)
      .def_readwrite ("weight",    &Linedata::weight)
      .def_readwrite ("frequency", &Linedata::frequency)
      .def_readwrite ("A",         &Linedata::A)
      .def_readwrite ("Ba",        &Linedata::Ba)
      .def_readwrite ("Bs",        &Linedata::Bs)
      .def_readwrite ("ncolpar",   &Linedata::ncolpar)
      .def_readwrite ("colpar",    &Linedata::colpar)
      // constructor
      .def (py::init<>())
      // functions
      .def ("read",                &Linedata::read)
      .def ("write",               &Linedata::write);

  // Colpartner
  py::class_<CollisionPartner> (module, "CollisionPartner")
      // attributes
      .def_readwrite ("num_col_partner", &CollisionPartner::num_col_partner)
      .def_readwrite ("orth_or_para_H2", &CollisionPartner::orth_or_para_H2)
      .def_readwrite ("ntmp",            &CollisionPartner::ntmp)
      .def_readwrite ("ncol",            &CollisionPartner::ncol)
      .def_readwrite ("icol",            &CollisionPartner::icol)
      .def_readwrite ("jcol",            &CollisionPartner::jcol)
      .def_readwrite ("tmp",             &CollisionPartner::tmp)
      .def_readwrite ("Ce",              &CollisionPartner::Ce)
      .def_readwrite ("Cd",              &CollisionPartner::Cd)
      // constructor
      .def (py::init<>())
      // functions
      .def ("read",                      &CollisionPartner::read)
      .def ("write",                     &CollisionPartner::write);

}
