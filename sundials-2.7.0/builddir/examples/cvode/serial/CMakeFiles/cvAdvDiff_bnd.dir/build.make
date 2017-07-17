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
include examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/depend.make

# Include the progress variables for this target.
include examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/progress.make

# Include the compile flags for this target's objects.
include examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/flags.make

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o: examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/flags.make
examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o: ../examples/cvode/serial/cvAdvDiff_bnd.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvAdvDiff_bnd.c

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvAdvDiff_bnd.c > CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.i

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial/cvAdvDiff_bnd.c -o CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.s

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.requires:

.PHONY : examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.requires

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.provides: examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.requires
	$(MAKE) -f examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/build.make examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.provides.build
.PHONY : examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.provides

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.provides.build: examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o


# Object files for target cvAdvDiff_bnd
cvAdvDiff_bnd_OBJECTS = \
"CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o"

# External object files for target cvAdvDiff_bnd
cvAdvDiff_bnd_EXTERNAL_OBJECTS =

examples/cvode/serial/cvAdvDiff_bnd: examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o
examples/cvode/serial/cvAdvDiff_bnd: examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/build.make
examples/cvode/serial/cvAdvDiff_bnd: src/cvode/libsundials_cvode.so.2.9.0
examples/cvode/serial/cvAdvDiff_bnd: src/nvec_ser/libsundials_nvecserial.so.2.7.0
examples/cvode/serial/cvAdvDiff_bnd: /usr/lib/x86_64-linux-gnu/librt.so
examples/cvode/serial/cvAdvDiff_bnd: examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable cvAdvDiff_bnd"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvAdvDiff_bnd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/build: examples/cvode/serial/cvAdvDiff_bnd

.PHONY : examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/build

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/requires: examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/cvAdvDiff_bnd.c.o.requires

.PHONY : examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/requires

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial && $(CMAKE_COMMAND) -P CMakeFiles/cvAdvDiff_bnd.dir/cmake_clean.cmake
.PHONY : examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/clean

examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/cvode/serial/CMakeFiles/cvAdvDiff_bnd.dir/depend

