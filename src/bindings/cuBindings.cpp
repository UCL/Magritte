#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;

#include "Simulation/Raypair/raypair.cuh"


PYBIND11_MODULE (gpuMagritte, module)
{
  py::class_<RayPair> (module, "RayPair")
      // attributes
      .def_readwrite ("n_ar",    &RayPair::n_ar)
      .def_readwrite ("n_r",     &RayPair::n_r)
      .def_readwrite ("first",   &RayPair::first)
      .def_readwrite ("last",    &RayPair::last)
      .def_readwrite ("ndep",    &RayPair::ndep)
      .def_readwrite ("I_bdy_0", &RayPair::I_bdy_0)
      .def_readwrite ("I_bdy_n", &RayPair::I_bdy_n)
      // constructor
      .def (py::init<>())
      // functions
  //    .def ("allocate",         &Model::allocate)
  //    .def ("copyHostToDevice", &Model::copyHostToDevice)
  //    .def ("compute",          &Model::compute)
  //    .def ("copyDeviceToHost", &Model::copyDeviceToHost)
  //    .def ("free",             &Model::free)
      .def ("solve",            &RayPair::solve);
}
