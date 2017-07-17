# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir

# Include any dependencies generated for this target.
include examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/depend.make

# Include the progress variables for this target.
include examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/progress.make

# Include the compile flags for this target's objects.
include examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/flags.make

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o: examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/flags.make
examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o: ../examples/ida/fcmix_serial/fidaRoberts_dns.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial && /usr/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/fcmix_serial/fidaRoberts_dns.f -o CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial && /usr/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/fcmix_serial/fidaRoberts_dns.f > CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.i

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial && /usr/bin/gfortran $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/fcmix_serial/fidaRoberts_dns.f -o CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.s

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.requires:

.PHONY : examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.requires

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.provides: examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.requires
	$(MAKE) -f examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/build.make examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.provides.build
.PHONY : examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.provides

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.provides.build: examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o


# Object files for target fidaRoberts_dns
fidaRoberts_dns_OBJECTS = \
"CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o"

# External object files for target fidaRoberts_dns
fidaRoberts_dns_EXTERNAL_OBJECTS =

examples/ida/fcmix_serial/fidaRoberts_dns: examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o
examples/ida/fcmix_serial/fidaRoberts_dns: examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/build.make
examples/ida/fcmix_serial/fidaRoberts_dns: src/ida/fcmix/libsundials_fida.a
examples/ida/fcmix_serial/fidaRoberts_dns: src/ida/libsundials_ida.so.2.9.0
examples/ida/fcmix_serial/fidaRoberts_dns: src/nvec_ser/libsundials_fnvecserial.so.2.7.0
examples/ida/fcmix_serial/fidaRoberts_dns: src/nvec_ser/libsundials_nvecserial.so.2.7.0
examples/ida/fcmix_serial/fidaRoberts_dns: /usr/lib/x86_64-linux-gnu/librt.so
examples/ida/fcmix_serial/fidaRoberts_dns: examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking Fortran executable fidaRoberts_dns"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fidaRoberts_dns.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/build: examples/ida/fcmix_serial/fidaRoberts_dns

.PHONY : examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/build

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/requires: examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/fidaRoberts_dns.f.o.requires

.PHONY : examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/requires

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial && $(CMAKE_COMMAND) -P CMakeFiles/fidaRoberts_dns.dir/cmake_clean.cmake
.PHONY : examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/clean

examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/fcmix_serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/ida/fcmix_serial/CMakeFiles/fidaRoberts_dns.dir/depend

