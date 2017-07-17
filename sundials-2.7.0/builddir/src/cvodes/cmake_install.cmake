# Install script for directory: /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/cvodes

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  MESSAGE("
Install CVODES
")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/cvodes/libsundials_cvodes.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsundials_cvodes.so.2.9.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsundials_cvodes.so.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsundials_cvodes.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/cvodes/libsundials_cvodes.so.2.9.0"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/cvodes/libsundials_cvodes.so.2"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/cvodes/libsundials_cvodes.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsundials_cvodes.so.2.9.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsundials_cvodes.so.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsundials_cvodes.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/cvodes" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_band.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_bandpre.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_bbdpre.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_dense.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_diag.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_direct.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_sparse.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_spbcgs.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_spgmr.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_spils.h"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/include/cvodes/cvodes_sptfqmr.h"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/cvodes" TYPE FILE FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/cvodes/cvodes_impl.h")
endif()

