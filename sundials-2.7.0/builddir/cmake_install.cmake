# Install script for directory: /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sundials" TYPE FILE FILES "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/include/sundials/sundials_config.h")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/sundials/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_ser/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/arkode/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/arkode/fcmix/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/cvode/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/cvode/fcmix/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/cvodes/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/idas/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/kinsol/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/kinsol/fcmix/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/arkode/C_serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/arkode/F77_serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/arkode/C_parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/arkode/F77_parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/fcmix_serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/fcmix_parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/kinsol/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/kinsol/fcmix_serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/kinsol/parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/kinsol/fcmix_parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/nvector/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/nvector/parallel/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
