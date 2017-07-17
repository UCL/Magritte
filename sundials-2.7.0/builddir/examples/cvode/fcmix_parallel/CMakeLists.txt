# -----------------------------------------------------------------
# $Revision: 4958 $
# $Date: 2016-09-23 14:02:13 -0700 (Fri, 23 Sep 2016) $
# -----------------------------------------------------------------
# Programmer: Radu Serban @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2007, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# -----------------------------------------------------------------
# CMakeLists.txt for CVODE Fortran parallel examples
#
# This file is generated from a template using  various variables
# set at configuration time. It can be used as a template for
# other user CMakeLists configuration files.
#
# Note: if the solver was successfully configured with Blas/Lapack
# support, the Blas/Lapack libraries are specified through the 
# variable LAPACK_LIBRARIES. Otherwise, this variable should contain
# an empty string or LAPACK_LIBRARIES-NOTFOUND. If this variable 
# contains a valid library entry, we add it to each example traget
# whether they use the Lapack module or not. This is done in order
# to address the case in which the SUNDIALS libraries are shared 
# objects. In that case, the solver library references Lapack 
# symbols which must be always resolved by linking against the
# Blas/Lapack libraries. If only static SUNDIALS libraries have 
# been built, it is not required to link the Blas/Lapack libraries
# for examples that do not use that module...
#
# -----------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)

# Specify project name
PROJECT(FCVODE_parallel_examples Fortran)

MARK_AS_ADVANCED(EXECUTABLE_OUTPUT_PATH LIBRARY_OUTPUT_PATH)

# Set the names of the examples to be built
SET(examples  fcvDiag_non_p fcvDiag_kry_bbd_p fcvDiag_kry_p)

# Set name of MPI compiler to be used (F77 or MPIF77)
SET(MPI_MPIF77 /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpif77 CACHE STRING "MPIF77 compiler script")
SET(CMAKE_Fortran_COMPILER ${MPI_MPIF77})

# Set search path for SUNDIALS libraries 
SET(SUNDIALS_LIB_DIR /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/lib)

# Find the SUNDIALS solver's library
FIND_LIBRARY(SUNDIALS_SOLVER_LIB
  sundials_cvode ${SUNDIALS_LIB_DIR}
  DOC "CVODE library")

FIND_LIBRARY(SUNDIALS_SOLVER_FLIB
  sundials_fcvode ${SUNDIALS_LIB_DIR}
  DOC "CVODE F77-C library")

# Find the NVECTOR library
FIND_LIBRARY(SUNDIALS_NVEC_LIB
  sundials_nvecparallel ${SUNDIALS_LIB_DIR}
  DOC "NVECTOR library")

FIND_LIBRARY(SUNDIALS_NVEC_FLIB
  sundials_fnvecparallel ${SUNDIALS_LIB_DIR}
  DOC "NVECTOR F77-C library")

# Set additional libraries
SET(SUNDIALS_EXTRA_LIB  -lm /usr/lib/x86_64-linux-gnu/librt.so CACHE STRING "Additional libraries")

# Set other libraries libraries
SET(LAPACK_LIBRARIES  CACHE STRING "Lapack libraries")
SET(SUPERLUMT_LIBRARIES  CACHE STRING "SuperLUMT libraries")
SET(KLU_LIBRARIES  CACHE STRING "KLU libraries")
SET(PETSC_LIBRARIES  CACHE STRING "PETSC libraries")

# List of all libraries
SET(SUNDIALS_LIBS 
  ${SUNDIALS_SOLVER_FLIB} ${SUNDIALS_SOLVER_LIB} 
  ${SUNDIALS_NVEC_FLIB} ${SUNDIALS_NVEC_LIB} 
  ${SUNDIALS_EXTRA_LIB})
IF(LAPACK_LIBRARIES)
  LIST(APPEND SUNDIALS_LIBS ${LAPACK_LIBRARIES})
ENDIF(LAPACK_LIBRARIES)

IF(SUPERLUMT_LIBRARIES)
  LIST(APPEND SUNDIALS_LIBS ${SUPERLUMT_LIBRARIES})
ENDIF(SUPERLUMT_LIBRARIES)

IF(KLU_LIBRARIES)
  LIST(APPEND SUNDIALS_LIBS ${KLU_LIBRARIES})
ENDIF(KLU_LIBRARIES)

IF(PETSC_LIBRARIES)
  LIST(APPEND SUNDIALS_LIBS ${PETSC_LIBRARIES})
ENDIF(PETSC_LIBRARIES)

# Build each example one by one
FOREACH(example ${examples})
  ADD_EXECUTABLE(${example} ${example}.f)
  TARGET_LINK_LIBRARIES(${example} ${SUNDIALS_LIBS}) 
ENDFOREACH(example ${examples})
