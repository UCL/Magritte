# CMakeLists.txt for Python Python Bindings
###########################################

# Point to the include directory
include_directories (../src)
include_directories (../dependencies)
include_directories (../dependencies/pybind11/include)

# pybind11 command to create the python module in bin directory
pybind11_add_module (magritte
  pyBindings.cpp
  ../src/Model/model.cpp
  ../src/Model/parameters.cpp
  ../src/Model/Geometry/geometry.cpp
  ../src/Model/Geometry/Cells/cells.cpp
  ../src/Model/Geometry/Rays/rays.cpp
  ../src/Model/Geometry/Boundary/boundary.cpp
  ../src/Model/Chemistry/chemistry.cpp
  ../src/Model/Chemistry/Species/species.cpp
  ../src/Model/Lines/lines.cpp
  ../src/Model/Lines/LineProducingSpecies/lineProducingSpecies.cpp
  ../src/Model/Lines/LineProducingSpecies/Quadrature/quadrature.cpp
  ../src/Model/Lines/LineProducingSpecies/Linedata/linedata.cpp
  ../src/Model/Lines/LineProducingSpecies/Linedata/CollisionPartner/collisionPartner.cpp
  ../src/Model/Radiation/radiation.cpp
  ../src/Model/Radiation/Frequencies/frequencies.cpp
  ../src/Model/Thermodynamics/thermodynamics.cpp
  ../src/Model/Thermodynamics/Temperature/temperature.cpp
  ../src/Model/Thermodynamics/Turbulence/turbulence.cpp
  ../src/Functions/heapsort.cpp
  ../src/Simulation/simulation.cpp)

# pybind11 command to create the python module in bin directory
pybind11_add_module (ioMagritte
  ioBindings.cpp
  ../src/Io/io.cpp
  ../src/Io/io_text.cpp
  ../src/Io/io_Python.cpp)
