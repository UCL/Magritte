// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "Tools/types.hpp"
#include "Model/model.hpp"
#include "Simulation/simulation.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
namespace py = pybind11;


PYBIND11_MAKE_OPAQUE (Long1);
PYBIND11_MAKE_OPAQUE (Long2);
PYBIND11_MAKE_OPAQUE (Long3);
PYBIND11_MAKE_OPAQUE (Long4);

PYBIND11_MAKE_OPAQUE (Double1);
PYBIND11_MAKE_OPAQUE (Double2);
PYBIND11_MAKE_OPAQUE (Double3);

PYBIND11_MAKE_OPAQUE (String1);

PYBIND11_MAKE_OPAQUE (std::vector<LineProducingSpecies>);
PYBIND11_MAKE_OPAQUE (std::vector<CollisionPartner>);


PYBIND11_MODULE (magritte, module)
{

  // Module docstring
  module.doc() = "Magritte module";


  // Define vector types
  py::bind_vector<Long1>   (module, "Long1");
  py::bind_vector<Long2>   (module, "Long2");
  py::bind_vector<Long3>   (module, "Long3");
  py::bind_vector<Long4>   (module, "Long4");

  py::bind_vector<Double1> (module, "Double1");
  py::bind_vector<Double2> (module, "Double2");
  py::bind_vector<Double3> (module, "Double3");

  py::bind_vector<String1> (module, "String1");

  py::bind_vector<Lambda1> (module, "Lambda1");
  py::bind_vector<Lambda2> (module, "Lambda2");

  py::bind_vector<std::vector<LineProducingSpecies>> (module, "vLineProducingSpecies");
  py::bind_vector<std::vector<CollisionPartner>>     (module, "vCollisionPartner");


  // Model
  py::class_<Model> (module, "Model")
      // attributes
      .def_readwrite ("parameters",     &Model::parameters)
      .def_readwrite ("geometry",       &Model::geometry)
      .def_readwrite ("chemistry",      &Model::chemistry)
      .def_readwrite ("lines",          &Model::lines)
      .def_readwrite ("thermodynamics", &Model::thermodynamics)
      .def_readwrite ("radiation",      &Model::radiation)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                     &Model::read)
      .def ("write",                    &Model::write);


  // Parameters
  py::class_<Parameters> (module, "Parameters")
      // constructor
      .def (py::init())
      .def_readwrite ("r",          &Parameters::r)
      .def_readwrite ("o",          &Parameters::o)
      .def_readwrite ("f",          &Parameters::o)
      .def_readwrite ("n_off_diag", &Parameters::n_off_diag)
      .def_readwrite ("max_width_fraction", &Parameters::max_width_fraction)
      // setters
      .def ("set_ncells",           &Parameters::set_ncells        )
      .def ("set_ncameras",         &Parameters::set_ncameras      )
      .def ("set_nrays",            &Parameters::set_nrays         )
      .def ("set_nrays",            &Parameters::set_nrays_red     )
      .def ("set_nboundary",        &Parameters::set_nboundary     )
      .def ("set_nfreqs",           &Parameters::set_nfreqs        )
      .def ("set_nfreqs_red",       &Parameters::set_nfreqs_red    )
      .def ("set_nspecs",           &Parameters::set_nspecs        )
      .def ("set_nlspecs",          &Parameters::set_nlspecs       )
      .def ("set_nlines",           &Parameters::set_nlines        )
      .def ("set_nquads",           &Parameters::set_nquads        )
      .def ("set_pop_prec",         &Parameters::set_pop_prec      )
      .def ("set_use_scattering",   &Parameters::set_use_scattering)
      // getters
      .def ("ncells",               &Parameters::ncells        )
      .def ("ncameras",             &Parameters::ncameras      )
      .def ("nrays",                &Parameters::nrays         )
      .def ("nrays_red",            &Parameters::nrays_red     )
      .def ("nboundary",            &Parameters::nboundary     )
      .def ("nfreqs",               &Parameters::nfreqs        )
      .def ("nfreqs_red",           &Parameters::nfreqs_red    )
      .def ("nspecs",               &Parameters::nspecs        )
      .def ("nlspecs",              &Parameters::nlspecs       )
      .def ("nlines",               &Parameters::nlines        )
      .def ("nquads",               &Parameters::nquads        )
      .def ("pop_prec",             &Parameters::pop_prec      )
      .def ("use_scattering",       &Parameters::use_scattering)
      // functions
      .def ("read",                 &Parameters::read      )
      .def ("write",                &Parameters::write     );



  // Geometry
  py::class_<Geometry> (module, "Geometry")
      // attributes
      .def_readwrite ("cells",    &Geometry::cells)
      .def_readwrite ("rays",     &Geometry::rays)
      .def_readwrite ("boundary", &Geometry::boundary)
      .def_readwrite ("cameras",  &Geometry::cameras)
      // constructor
      .def (py::init())
      // functions
      .def ("read",               &Geometry::read)
      .def ("write",              &Geometry::write);


  // Cameras
  py::class_<Cameras> (module, "Cameras")
      // attributes
      .def_readwrite ("camera2cell_nr", &Cameras::camera2cell_nr)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                     &Cameras::read)
      .def ("write",                    &Cameras::write);


  // Cells
  py::class_<Cells> (module, "Cells")
      // attributes
      .def_readwrite ("x",           &Cells::x)
      .def_readwrite ("y",           &Cells::y)
      .def_readwrite ("z",           &Cells::z)
      .def_readwrite ("vx",          &Cells::vx)
      .def_readwrite ("vy",          &Cells::vy)
      .def_readwrite ("vz",          &Cells::vz)
      .def_readwrite ("n_neighbors", &Cells::n_neighbors)
      .def_readwrite ("neighbors",   &Cells::neighbors)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                  &Cells::read)
      .def ("write",                 &Cells::write);


  // Rays
  py::class_<Rays> (module, "Rays")
      // attributes
      .def_readwrite ("x",       &Rays::x)
      .def_readwrite ("y",       &Rays::y)
      .def_readwrite ("z",       &Rays::z)
      .def_readwrite ("weights", &Rays::weights)
      .def_readwrite ("antipod", &Rays::antipod)
      // constructor
      .def (py::init())
      // functions
      .def ("read",              &Rays::read)
      .def ("write",             &Rays::write);


  // Boundary
  py::class_<Boundary> (module, "Boundary")
      // attributes
      .def_readwrite ("boundary2cell_nr", &Boundary::boundary2cell_nr)
      .def_readwrite ("cell2boundary_nr", &Boundary::cell2boundary_nr)
      .def_readwrite ("boundary",         &Boundary::boundary)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                       &Boundary::read)
      .def ("write",                      &Boundary::write);


  // Thermodynamics
  py::class_<Thermodynamics> (module, "Thermodynamics")
      // attributes
      .def_readwrite ("temperature", &Thermodynamics::temperature)
      .def_readwrite ("turbulence",  &Thermodynamics::turbulence)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                  &Thermodynamics::read)
      .def ("write",                 &Thermodynamics::write);


  // Temperature
  py::class_<Temperature> (module, "Temperature")
      // attributes
      .def_readwrite ("gas", &Temperature::gas)
      // constructor
      .def (py::init())
      // functions
      .def ("read",          &Temperature::read)
      .def ("write",         &Temperature::write);


  // Turbulence
  py::class_<Turbulence> (module, "Turbulence")
      // attributes
      .def_readwrite ("vturb2", &Turbulence::vturb2)
      // constructor
      .def (py::init())
      // functions
      .def ("read",             &Turbulence::read)
      .def ("write",            &Turbulence::write);


  // Chemistry
  py::class_<Chemistry> (module, "Chemistry")
      // attributes
      .def_readwrite ("species", &Chemistry::species)
      // constructor
      .def (py::init())
      // functions
      .def ("read",              &Chemistry::read)
      .def ("write",             &Chemistry::write);


  // Species
  py::class_<Species> (module, "Species")
      // attributes
      .def_readwrite ("abundance", &Species::abundance)
      .def_readwrite ("sym",       &Species::sym)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                &Species::read)
      .def ("write",               &Species::write);


  // Lines
  py::class_<Lines> (module, "Lines")
      // attributes
      .def_readwrite ("lineProducingSpecies", &Lines::lineProducingSpecies)
      .def_readwrite ("emissivity",           &Lines::emissivity)
      .def_readwrite ("opacity",              &Lines::opacity)
      .def_readwrite ("line",                 &Lines::line)
      .def_readwrite ("line_index",           &Lines::line_index)
      // constructor
      .def (py::init<>())
      // functions
      .def ("read",                           &Lines::read)
      .def ("write",                          &Lines::write)
      .def ("set_emissivity_and_opacity",     &Lines::set_emissivity_and_opacity);


  // LineProducingSpecies
  py::class_<LineProducingSpecies> (module, "LineProducingSpecies")
      // attributes
      .def_readwrite ("linedata",         &LineProducingSpecies::linedata)
      .def_readwrite ("quadrature",       &LineProducingSpecies::quadrature)
      .def_readwrite ("Lambda",           &LineProducingSpecies::lambda)
      .def_readwrite ("Jeff",             &LineProducingSpecies::Jeff)
      .def_readwrite ("Jlin",             &LineProducingSpecies::Jlin)
      .def_readwrite ("nr_line",          &LineProducingSpecies::nr_line)
      .def_readwrite ("population",       &LineProducingSpecies::population)
      .def_readwrite ("population_tot",   &LineProducingSpecies::population_tot)
      .def_readwrite ("population_prev1", &LineProducingSpecies::population_prev1)
      .def_readwrite ("population_prev2", &LineProducingSpecies::population_prev2)
      .def_readwrite ("population_prev3", &LineProducingSpecies::population_prev3)
      .def_readwrite ("ncells",           &LineProducingSpecies::ncells)
      // constructor
      .def (py::init<>())
      // functions
      .def ("read",                       &LineProducingSpecies::read)
      .def ("write",                      &LineProducingSpecies::write);


  // Lambda
  py::class_<Lambda> (module, "Lambda")
      // attributes
      .def_readwrite ("Ls", &Lambda::Ls)
      .def_readwrite ("nr", &Lambda::nr)
      // constructor
      .def (py::init<>())
      // functions
      .def ("add_entry",    &Lambda::add_entry);


  // Quadrature
  py::class_<Quadrature> (module, "Quadrature")
      // attributes
      .def_readwrite ("roots",   &Quadrature::roots)
      .def_readwrite ("weights", &Quadrature::weights)
      // constructor
      .def (py::init<>())
      // functions
      .def ("read",              &Quadrature::read)
      .def ("write",             &Quadrature::write);


  // Linedata
  py::class_<Linedata> (module, "Linedata")
      // attributes
      .def_readwrite ("num",          &Linedata::num)
      .def_readwrite ("sym",          &Linedata::sym)
      .def_readwrite ("inverse_mass", &Linedata::inverse_mass)
      .def_readwrite ("nlev",         &Linedata::nlev)
      .def_readwrite ("nrad",         &Linedata::nrad)
      .def_readwrite ("irad",         &Linedata::irad)
      .def_readwrite ("jrad",         &Linedata::jrad)
      .def_readwrite ("energy",       &Linedata::energy)
      .def_readwrite ("weight",       &Linedata::weight)
      .def_readwrite ("frequency",    &Linedata::frequency)
      .def_readwrite ("A",            &Linedata::A)
      .def_readwrite ("Ba",           &Linedata::Ba)
      .def_readwrite ("Bs",           &Linedata::Bs)
      .def_readwrite ("ncolpar",      &Linedata::ncolpar)
      .def_readwrite ("colpar",       &Linedata::colpar)
      // constructor
      .def (py::init<>())
      // functions
      .def ("read",                   &Linedata::read)
      .def ("write",                  &Linedata::write);


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


  // Radiation
  py::class_<Radiation> (module, "Radiation")
      // attributes
      .def_readwrite ("frequencies", &Radiation::frequencies)
      .def_readwrite ("u",           &Radiation::u)
      .def_readwrite ("v",           &Radiation::v)
      .def_readwrite ("J",           &Radiation::J)
      // constructor
      .def (py::init())
      // functions
      .def ("read",                  &Radiation::read)
      .def ("write",                 &Radiation::write);


  // Frequencies
  py::class_<Frequencies> (module, "Frequencies")
      // attributes
      .def_readwrite ("nu", &Frequencies::nu)
      // constructor
      .def (py::init())
      // functions
      .def ("read",         &Frequencies::read)
      .def ("write",        &Frequencies::write);




  // Simulation
  py::class_<Simulation, Model> (module, "Simulation")
      // constructor
      .def (py::init<>())
      // attributes
      .def_readonly ("error_max",              &Simulation::error_max)
      .def_readonly ("error_mean",             &Simulation::error_mean)
      //.def_readonly ("rayPair",                &Simulation::rayPair)

      // functions
      .def ("compute_spectral_discretisation", &Simulation::compute_spectral_discretisation)
      .def ("compute_boundary_intensities",    &Simulation::compute_boundary_intensities)
      .def ("compute_LTE_level_populations",   &Simulation::compute_LTE_level_populations)
      .def ("compute_radiation_field",         &Simulation::compute_radiation_field)
      .def ("compute_and_write_image",         &Simulation::compute_and_write_image)
      .def ("compute_level_populations",       &Simulation::compute_level_populations)
      .def ("compute_level_populations_opts",  &Simulation::compute_level_populations_opts);

  // RayPair
  py::class_<RayPair> (module, "RayPair")
      // constructor
      .def (py::init<>())
      // attributes
      .def_readonly ("n_ar", &RayPair::n_ar)
      .def_readonly ("n_r",  &RayPair::n_r)
      .def_readonly ("ndep", &RayPair::ndep);
      // .def_readonly ("chi",  &RayPair::chi)
      // .def_readonly ("Su",   &RayPair::Su)
      // .def_readonly ("Sv",   &RayPair::Sv)
      // .def_readonly ("nrs",  &RayPair::nrs)
      // .def_readonly ("frs",  &RayPair::frs)
      // .def_readonly ("dtau", &RayPair::dtau)

      // functions

  // Image
  py::class_<Image> (module, "Image")
      // attributes
      .def_readonly ("ImX", &Image::ImX)
      .def_readonly ("ImY", &Image::ImY)
      .def_readonly ("I_p", &Image::I_p)
      .def_readonly ("I_m", &Image::I_m)
      // constructor
      .def (py::init<const long, const Parameters &>())
      // functions
      .def ("write", &Image::write);

}
