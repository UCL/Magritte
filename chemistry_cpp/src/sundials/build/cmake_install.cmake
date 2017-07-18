# Install script for directory: /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials")
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

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sundials" TYPE FILE FILES "/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/include/sundials/sundials_config.h")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/src/sundials/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/src/nvec_ser/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/src/nvec_par/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/src/nvec_openmp/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/src/cvode/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/nvector/serial/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/nvector/parallel/cmake_install.cmake")
  include("/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/nvector/C_openmp/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
