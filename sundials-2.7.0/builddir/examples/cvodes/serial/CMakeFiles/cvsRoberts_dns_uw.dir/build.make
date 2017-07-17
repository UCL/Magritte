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
include examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/depend.make

# Include the progress variables for this target.
include examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/progress.make

# Include the compile flags for this target's objects.
include examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/flags.make

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o: examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/flags.make
examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o: ../examples/cvodes/serial/cvsRoberts_dns_uw.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvodes/serial/cvsRoberts_dns_uw.c

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvodes/serial/cvsRoberts_dns_uw.c > CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.i

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvodes/serial/cvsRoberts_dns_uw.c -o CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.s

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.requires:

.PHONY : examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.requires

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.provides: examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.requires
	$(MAKE) -f examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/build.make examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.provides.build
.PHONY : examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.provides

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.provides.build: examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o


# Object files for target cvsRoberts_dns_uw
cvsRoberts_dns_uw_OBJECTS = \
"CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o"

# External object files for target cvsRoberts_dns_uw
cvsRoberts_dns_uw_EXTERNAL_OBJECTS =

examples/cvodes/serial/cvsRoberts_dns_uw: examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o
examples/cvodes/serial/cvsRoberts_dns_uw: examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/build.make
examples/cvodes/serial/cvsRoberts_dns_uw: src/cvodes/libsundials_cvodes.so.2.9.0
examples/cvodes/serial/cvsRoberts_dns_uw: src/nvec_ser/libsundials_nvecserial.so.2.7.0
examples/cvodes/serial/cvsRoberts_dns_uw: /usr/lib/x86_64-linux-gnu/librt.so
examples/cvodes/serial/cvsRoberts_dns_uw: examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable cvsRoberts_dns_uw"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvsRoberts_dns_uw.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/build: examples/cvodes/serial/cvsRoberts_dns_uw

.PHONY : examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/build

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/requires: examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/cvsRoberts_dns_uw.c.o.requires

.PHONY : examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/requires

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial && $(CMAKE_COMMAND) -P CMakeFiles/cvsRoberts_dns_uw.dir/cmake_clean.cmake
.PHONY : examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/clean

examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvodes/serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/cvodes/serial/CMakeFiles/cvsRoberts_dns_uw.dir/depend

