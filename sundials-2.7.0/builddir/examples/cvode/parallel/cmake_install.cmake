# Install script for directory: /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel

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
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvAdvDiff_diag_p.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvAdvDiff_diag_p.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvAdvDiff_diag_p.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvAdvDiff_diag_p.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvAdvDiff_non_p.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvAdvDiff_non_p.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvAdvDiff_non_p.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvAdvDiff_non_p.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvDiurnal_kry_bbd_p.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvDiurnal_kry_bbd_p.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvDiurnal_kry_bbd_p.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvDiurnal_kry_bbd_p.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvDiurnal_kry_p.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/cvDiurnal_kry_p.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvDiurnal_kry_p.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/cvDiurnal_kry_p.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/README")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel" TYPE FILE FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/parallel/README")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/CMakeLists.txt")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel" TYPE FILE FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/parallel/CMakeLists.txt")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel/Makefile")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/parallel" TYPE FILE RENAME "Makefile" FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/parallel/Makefile_ex")
endif()

