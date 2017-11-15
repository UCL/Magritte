# CMakeLists.txt for Magritte setup
# ---------------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Create setup library

add_library( _setup
setup_data_structures.cpp
setup_initializers.cpp
setup_data_tools.cpp
setup_healpixvectors.cpp )


# Add the HEALPix directory

add_subdirectory(HEALPix)


# Create executable for setup

add_executable( setup.exe   setup.cpp )


# Link the setup library to the executable

target_link_libraries( setup.exe   _setup _HEALPix )
