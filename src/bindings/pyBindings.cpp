// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "configure.hpp"
#include "Tools/types.hpp"
#include "Io/cpp/io_cpp_text.hpp"
#include "Io/python/io_python.hpp"
#include "Model/model.hpp"
#include "Simulation/simulation.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
namespace py = pybind11;


PYBIND11_MAKE_OPAQUE (std::vector<LineProducingSpecies>);
PYBIND11_MAKE_OPAQUE (std::vector<CollisionPartner>);


PYBIND11_MODULE (core, module)
{

    // Module docstring
    module.doc() = "Magritte core module containing the C++ library.";


    // Define vector types
    py::bind_vector<std::vector<LineProducingSpecies>> (module, "vLineProducingSpecies");
    py::bind_vector<std::vector<CollisionPartner>>     (module, "vCollisionPartner");


    // Io, base class
    py::class_<Io> (module, "Io");

    // IoText
    py::class_<IoText, Io> (module, "IoText")
        // attributes
        .def_readonly ("io_file", &IoText::io_file)
        // constructor
        .def (py::init<const string &>());

#   if (PYTHON_IO)
        // IoPython
        py::class_<IoPython, Io> (module, "IoPython")
            // attributes
            .def_readonly ("implementation", &IoPython::implementation)
            .def_readonly ("io_file",        &IoPython::io_file)
            // constructor
            .def (py::init<const string &, const string &>())
            .def ("read_number", (int (IoPython::*)(const string, size_t&) const) &IoPython::read_number);
#   endif


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
        .def_readwrite ("n_off_diag",         &Parameters::n_off_diag)
        .def_readwrite ("max_width_fraction", &Parameters::max_width_fraction)
        // setters
        .def ("set_ncells",                   &Parameters::set_ncells              )
        .def ("set_nrays",                    &Parameters::set_nrays               )
        .def ("set_nrays_red",                &Parameters::set_nrays_red           )
        .def ("set_order_min",                &Parameters::set_order_min           )
        .def ("set_order_max",                &Parameters::set_order_max           )
        .def ("set_nboundary",                &Parameters::set_nboundary           )
        .def ("set_nfreqs",                   &Parameters::set_nfreqs              )
        .def ("set_nfreqs_red",               &Parameters::set_nfreqs_red          )
        .def ("set_nspecs",                   &Parameters::set_nspecs              )
        .def ("set_nlspecs",                  &Parameters::set_nlspecs             )
        .def ("set_nlines",                   &Parameters::set_nlines              )
        .def ("set_nquads",                   &Parameters::set_nquads              )
        .def ("set_pop_prec",                 &Parameters::set_pop_prec            )
        .def ("set_use_scattering",           &Parameters::set_use_scattering      )
        .def ("set_spherical_symmetry",       &Parameters::set_spherical_symmetry  )
        .def ("set_adaptive_ray_tracing",     &Parameters::set_adaptive_ray_tracing)
        // getters
        .def ("ncells",                       &Parameters::ncells              )
        .def ("nrays",                        &Parameters::nrays               )
        .def ("nrays_red",                    &Parameters::nrays_red           )
        .def ("order_min",                    &Parameters::order_min           )
        .def ("order_max",                    &Parameters::order_max           )
        .def ("nboundary",                    &Parameters::nboundary           )
        .def ("nfreqs",                       &Parameters::nfreqs              )
        .def ("nfreqs_red",                   &Parameters::nfreqs_red          )
        .def ("nspecs",                       &Parameters::nspecs              )
        .def ("nlspecs",                      &Parameters::nlspecs             )
        .def ("nlines",                       &Parameters::nlines              )
        .def ("nquads",                       &Parameters::nquads              )
        .def ("pop_prec",                     &Parameters::pop_prec            )
        .def ("use_scattering",               &Parameters::use_scattering      )
        .def ("spherical_symmetry",           &Parameters::spherical_symmetry  )
        .def ("adaptive_ray_tracing",         &Parameters::adaptive_ray_tracing)
        // functions
        .def ("read",                         &Parameters::read                )
        .def ("write",                        &Parameters::write               );



    // Geometry
    py::class_<Geometry> (module, "Geometry")
        // attributes
        .def_readwrite ("cells",    &Geometry::cells)
        .def_readwrite ("rays",     &Geometry::rays)
        .def_readwrite ("boundary", &Geometry::boundary)
        // constructor
        .def (py::init())
        // functions
        .def ("set_adaptive_rays",  &Geometry::set_adaptive_rays)
        .def ("read",               &Geometry::read)
        .def ("write",              &Geometry::write);


    // Cells
    py::class_<Cells> (module, "Cells")
        // attributes
        .def_readwrite ("position",    &Cells::position)
        .def_readwrite ("velocity",    &Cells::velocity)
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
        .def_readwrite ("rays",       &Rays::rays)
        .def_readwrite ("weights",    &Rays::weights)
        .def_readwrite ("antipod",    &Rays::antipod)
        .def_readwrite ("nrays",      &Rays::nrays)
        .def_readwrite ("half_nrays", &Rays::half_nrays)
        .def_readwrite ("dir",        &Rays::dir)
        .def_readwrite ("wgt",        &Rays::wgt)
        // constructor
        .def (py::init())
        // functions
        .def ("ray",                  &Rays::ray)
        .def ("weight",               &Rays::weight)
        .def ("get_order",            &Rays::get_order)
        .def ("get_pixel",            &Rays::get_pixel)
        .def ("read",                 &Rays::read)
        .def ("write",                &Rays::write);


    py::enum_<BoundaryCondition>(module, "BoundaryCondition")
        .value("Zero",    Zero)
        .value("Thermal", Thermal)
        .value("CMB",     CMB)
        .export_values();


    // Boundary
    py::class_<Boundary> (module, "Boundary")
        // attributes
        .def_readwrite ("boundary2cell_nr",     &Boundary::boundary2cell_nr)
        .def_readwrite ("cell2boundary_nr",     &Boundary::cell2boundary_nr)
        .def_readwrite ("boundary",             &Boundary::boundary)
        .def_readwrite ("boundary_condition",   &Boundary::boundary_condition)
        .def_readwrite ("boundary_temperature", &Boundary::boundary_temperature)
        // constructor
        .def (py::init())
        // functions
        .def ("read",                           &Boundary::read)
        .def ("write",                          &Boundary::write);


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
        .def_readwrite ("Lambda",           &LineProducingSpecies::lambda) // "lambda" is invalid in Python, use "Lambda"
        .def_readwrite ("Jeff",             &LineProducingSpecies::Jeff)
        .def_readwrite ("Jlin",             &LineProducingSpecies::Jlin)
        .def_readwrite ("nr_line",          &LineProducingSpecies::nr_line)
        .def_readwrite ("population",       &LineProducingSpecies::population)
        .def_readwrite ("population_tot",   &LineProducingSpecies::population_tot)
        .def_readwrite ("population_prev1", &LineProducingSpecies::population_prev1)
        .def_readwrite ("population_prev2", &LineProducingSpecies::population_prev2)
        .def_readwrite ("population_prev3", &LineProducingSpecies::population_prev3)
        .def_readwrite ("ncells",           &LineProducingSpecies::ncells)
        .def_readwrite ("RT",               &LineProducingSpecies::RT)
        .def_readwrite ("LambdaStar",       &LineProducingSpecies::LambdaStar)
        .def_readwrite ("LambdaTest",       &LineProducingSpecies::LambdaTest)
        // constructor
        .def (py::init<>())
        // functions
        .def ("read",                       &LineProducingSpecies::read)
        .def ("write",                      &LineProducingSpecies::write)
        .def ("print_populations",          &LineProducingSpecies::print_populations)
        .def ("index",                      &LineProducingSpecies::index);


    // Lambda
    py::class_<Lambda> (module, "Lambda")
        // attributes
        .def_readwrite ("Ls",   &Lambda::Ls)
        .def_readwrite ("nr",   &Lambda::nr)
        .def_readwrite ("size", &Lambda::size)
        .def_readwrite ("Lss",  &Lambda::Lss)
        .def_readwrite ("nrs",  &Lambda::nrs)
        // constructor
        .def (py::init<>())
        // functions
        .def ("add_element",    &Lambda::add_element)
        .def ("linearize_data", &Lambda::linearize_data)
        .def ("MPI_gather",    &Lambda::MPI_gather);


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
        .def_readwrite ("I_bdy",       &Radiation::I_bdy)
        .def_readwrite ("J",           &Radiation::J)
        // constructor
        .def (py::init())
        // functions
        .def ("read",                  &Radiation::read)
        .def ("write",                 &Radiation::write)
        .def ("get_u",                 &Radiation::get_u)
        .def ("get_v",                 &Radiation::get_v)
        .def ("get_J",                 &Radiation::get_J);


    // Frequencies
    py::class_<Frequencies> (module, "Frequencies")
        // attributes
        .def_readwrite ("nu", &Frequencies::nu)
        // constructor
        .def (py::init())
        // functions
        .def ("read",         &Frequencies::read)
        .def ("write",        &Frequencies::write)
        .def ("get_nu",       &Frequencies::get_nu);


//    // Solver
//    py::class_<Solver> (module, "Solver");
//
//    // gpuSolver
//    py::class_<gpuSolver, Solver> (module, "gpuSolver")
//    // constructor
//    .def (py::init<const Size &, const Size &, const Size &, const Size &, const Size &>());


    // Simulation
    py::class_<Simulation, Model> (module, "Simulation")
        // constructor
        .def (py::init<>())
        // attributes
        .def_readonly ("error_max",                    &Simulation::error_max)
        .def_readonly ("error_mean",                   &Simulation::error_mean)
        .def_readonly ("dtaus",                        &Simulation::dtaus)
        .def_readonly ("dZs",                          &Simulation::dZs)
        .def_readonly ("chis",                         &Simulation::chis)
        .def_readonly ("pre",                          &Simulation::pre)
        .def_readonly ("pos",                          &Simulation::pos)
//        .def_readonly ("Ld",                           &Simulation::Ld)
//        .def_readonly ("Lu",                           &Simulation::Lu)
//        .def_readonly ("Ll",                           &Simulation::Ll)
//        .def_readonly ("Lambda",                       &Simulation::Lambda)
        //.def_readonly ("rayPair",                      &Simulation::rayPair)
        .def_readonly ("timings",                      &Simulation::timings)
        .def_readonly ("nrpairs",                      &Simulation::nrpairs)
        .def_readonly ("depths",                       &Simulation::depths)
#       if (GPU_ACCELERATION)
        .def ("gpu_get_device_properties",             &Simulation::gpu_get_device_properties)
        .def ("gpu_compute_radiation_field",           &Simulation::gpu_compute_radiation_field)
        .def ("compute_radiation_field_gpu",           &Simulation::compute_radiation_field_gpu)
        .def ("gpu_compute_radiation_field_2",         &Simulation::gpu_compute_radiation_field_2)
#       endif
//        .def ("cpu_compute_radiation_field_2",         &Simulation::cpu_compute_radiation_field_2)
        .def ("cpu_compute_radiation_field",           &Simulation::cpu_compute_radiation_field)
        .def ("compute_radiation_field_cpu",           &Simulation::compute_radiation_field_cpu)
        .def ("compute_Jeff",                          &Simulation::compute_Jeff)
        .def ("compute_level_populations_from_stateq", &Simulation::compute_level_populations_from_stateq)
        // functions
        .def ("compute_spectral_discretisation",       &Simulation::compute_spectral_discretisation)
        .def ("compute_spectral_discretisation_image", &Simulation::compute_spectral_discretisation_image)
        .def ("compute_boundary_intensities",          (int(Simulation::*)(const Double1&))
                                                       &Simulation::compute_boundary_intensities)
        .def ("compute_boundary_intensities",          (int(Simulation::*)(void))
                                                       &Simulation::compute_boundary_intensities)
        .def ("compute_LTE_level_populations",         &Simulation::compute_LTE_level_populations)
        .def ("compute_radiation_field",               &Simulation::compute_radiation_field)
        .def ("compute_and_write_image",               &Simulation::compute_and_write_image)
        .def ("compute_level_populations",             (long(Simulation::*)(const Io&))
                                                       &Simulation::compute_level_populations)
        .def ("compute_level_populations",             (long(Simulation::*)(const Io&, const bool, const long))
                                                       &Simulation::compute_level_populations)
        .def("get_npoints_on_rays_comoving",           &Simulation::get_npoints_on_rays<CoMoving>)
        .def("get_npoints_on_rays_rest",               &Simulation::get_npoints_on_rays<Rest>);

//    // RayPair
//    py::class_<RayPair> (module, "RayPair")
//        // constructor
//        .def (py::init<>())
//        // attributes
//        .def_readwrite ("I_bdy_0", &RayPair::I_bdy_0)
//        .def_readwrite ("I_bdy_n", &RayPair::I_bdy_n)
//        .def_readonly ("n_ar",     &RayPair::n_ar)
//        .def_readonly ("n_r",      &RayPair::n_r)
//        .def_readonly ("ndep",     &RayPair::ndep)
//        .def_readonly ("A",        &RayPair::A)
//        .def_readonly ("C",        &RayPair::C)
//        .def_readonly ("Su",       &RayPair::Su)
//        .def_readonly ("Sv",       &RayPair::Sv)
//        .def_readonly ("dtau",     &RayPair::dtau)
//        .def_readonly ("L_diag",   &RayPair::L_diag)
//        .def_readonly ("L_upper",  &RayPair::L_upper)
//        .def_readonly ("L_lower",  &RayPair::L_lower)
//        // functions
//        .def ("resize",            &RayPair::resize)
//        .def ("initialize",        &RayPair::initialize)
//        .def ("set_term1_and_term2",
//              (void (RayPair::*)(const vReal&, const vReal&, const long))
//                                   &RayPair::set_term1_and_term2)
//        .def ("set_dtau",          &RayPair::set_dtau)
//        .def ("solve",             &RayPair::solve);


//    // RayBlock
//    py::class_<RayBlock> (module, "RayBlock")
//        // constructor
//        .def (py::init<>());
//


    // ProtoRayBlock
//    py::class_<ProtoRayBlock> (module, "ProtoRayBlock")
        // constructor
//        .def (py::init<>());

    // RayQueue
//    py::class_<RayQueue> (module, "RayQueue")
        // constructor
//        .def (py::init<>());


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
