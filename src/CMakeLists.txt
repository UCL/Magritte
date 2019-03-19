
# CMakeLists.txt for Magritte's source
######################################

# Write configuration file
configure_file (configure.hpp.in
  ${CMAKE_SOURCE_DIR}/src/configure.hpp)

# Headers are included in this folder
include_directories (.)

# Create Magritte library
add_library (_Magritte
  Io/io.cpp
  Io/io_text.cpp
  Model/model.cpp
  Model/parameters.cpp
  Model/Geometry/geometry.cpp
  Model/Geometry/Cells/cells.cpp
  Model/Geometry/Rays/rays.cpp
  Model/Geometry/Boundary/boundary.cpp
  Model/Chemistry/chemistry.cpp
  Model/Chemistry/Species/species.cpp
  Model/Lines/lines.cpp
  Model/Lines/LineProducingSpecies/lineProducingSpecies.cpp
  Model/Lines/LineProducingSpecies/Quadrature/quadrature.cpp
  Model/Lines/LineProducingSpecies/Linedata/linedata.cpp
  Model/Lines/LineProducingSpecies/Linedata/CollisionPartner/collisionPartner.cpp
  Model/Radiation/radiation.cpp
  Model/Radiation/Frequencies/frequencies.cpp
  Model/Thermodynamics/thermodynamics.cpp
  Model/Thermodynamics/Temperature/temperature.cpp
  Model/Thermodynamics/Turbulence/turbulence.cpp
  Functions/heapsort.cpp
  Simulation/simulation.cpp)

add_library (PyIo
  Io/io.cpp
  Io/io_Python.cpp)
target_link_libraries (PyIo PRIVATE
  pybind11::embed)
