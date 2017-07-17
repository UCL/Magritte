# Install script for directory: /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial

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
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvAdvDiff_bnd.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvAdvDiff_bnd.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvAdvDiff_bnd.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvAdvDiff_bnd.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvDirectDemo_ls.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvDirectDemo_ls.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvDirectDemo_ls.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvDirectDemo_ls.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvDiurnal_kry_bp.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvDiurnal_kry_bp.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvDiurnal_kry_bp.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvDiurnal_kry_bp.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvDiurnal_kry.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvDiurnal_kry.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvDiurnal_kry.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvDiurnal_kry.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvKrylovDemo_ls.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvKrylovDemo_ls.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvKrylovDemo_ls.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvKrylovDemo_ls.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvKrylovDemo_prec.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvKrylovDemo_prec.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvKrylovDemo_prec.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvKrylovDemo_prec.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvRoberts_dns.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvRoberts_dns.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvRoberts_dns.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvRoberts_dns.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvRoberts_dns_uw.c;/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/cvRoberts_dns_uw.out")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvRoberts_dns_uw.c"
    "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvRoberts_dns_uw.out"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/README")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/README")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/CMakeLists.txt")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial/CMakeLists.txt")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial/Makefile")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/instdir/examples/cvode/serial" TYPE FILE RENAME "Makefile" FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial/Makefile_ex")
endif()

