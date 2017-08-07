# CMakeLists.txt for 3D-RT_tests
# ------------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Specify project name

project(3D-RT_tests)


# Add compiler flags

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}" )
endif()


# Add 3DRT and HEALPix libraries containing the source code (which is to be tested)

find_library( 3DRT
              lib_3DRT.a /home/frederik/Dropbox/Astro/3D-RT/src
              DOC "3DRT library" )

find_library( HEALPix
              lib_HEALPix.a /home/frederik/Dropbox/Astro/3D-RT/src/HEALPix
              DOC "HEALPix library" )


# Add main library

add_library( _main   test_main.cpp  )


# Create executable to test exact_feautrier

add_executable( test_exact_feautrier   test_exact_feautrier.cpp )


# Link the necessary libraries

target_link_libraries( test_exact_feautrier   _main   ${3DRT}   ${HEALPix} )
