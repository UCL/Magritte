# CMakeLists.txt for Magritte
# ---------------------------


# Specify the minimum required CMake version
cmake_minimum_required(VERSION 2.8)


# Specify projact name
project(Magritte)


# Set build type to Release to get optimal performance
set(CMAKE_BUILD_TYPE Release)


# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=medium -std=c++11")


# Specify the version of the project
set(MAGRITTE_VERSION_MAJOR 0)
set(MAGRITTE_VERSION_MINOR 1)


# Resize the stack to 128 Mb (UNIX version)
#add_custom_target( RESIZE_STACK ALL
#                   COMMAND ulimit -S -s 131072 )


# Run setup script (if parameters.hpp changed)
add_custom_command(OUTPUT  src/Magritte_config.hpp
                   COMMAND bash run_setup.sh $(PROJECT_FOLDER)
                   WORKING_DIRECTORY pySetup
                   DEPENDS $(PROJECT_FOLDER)parameters.hpp)

add_custom_target(MAKE_SETUP
                  DEPENDS src/Magritte_config.hpp)


# Specify the subdirectory containing the source code
add_subdirectory(src)


# VTK includes
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


# Add compiler flags
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS
  "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -mcmodel=medium")
  set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mcmodel=medium")
  set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -mcmodel=medium")
endif()


# Create executable for Magritte main code
add_executable(Magritte.exe src/Magritte.cpp)
# Ensure that Magritte depends on setup command defined above
add_dependencies(Magritte.exe MAKE_SETUP)
# Link the library of the /src directory to Magritte main code
target_link_libraries(Magritte.exe _MAGRITTE)


# # Create executable for Magritte_io
# add_executable(Magritte_io.exe src/Magritte_io.cpp)
# # Ensure that Magritte_io depends on setup command defined above
# add_dependencies(Magritte_io.exe MAKE_SETUP)
# # Link the library of the /src directory to Magritte main code
# target_link_libraries(Magritte_io.exe _IO)
#
#
# # Create executable for Magritte_io
# add_executable(Magritte_o.exe src/Magritte_o.cpp)
# # Ensure that Magritte_io depends on setup command defined above
# add_dependencies(Magritte_o.exe MAKE_SETUP)
# # Link the library of the /src directory to Magritte main code
# target_link_libraries(Magritte_o.exe _O)
