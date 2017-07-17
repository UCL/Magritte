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
include examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/depend.make

# Include the progress variables for this target.
include examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/progress.make

# Include the compile flags for this target's objects.
include examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/flags.make

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o: examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/flags.make
examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o: ../examples/ida/parallel/idaFoodWeb_kry_p.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/parallel/idaFoodWeb_kry_p.c

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/parallel/idaFoodWeb_kry_p.c > CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.i

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/parallel/idaFoodWeb_kry_p.c -o CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.s

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.requires:

.PHONY : examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.requires

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.provides: examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.requires
	$(MAKE) -f examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/build.make examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.provides.build
.PHONY : examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.provides

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.provides.build: examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o


# Object files for target idaFoodWeb_kry_p
idaFoodWeb_kry_p_OBJECTS = \
"CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o"

# External object files for target idaFoodWeb_kry_p
idaFoodWeb_kry_p_EXTERNAL_OBJECTS =

examples/ida/parallel/idaFoodWeb_kry_p: examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o
examples/ida/parallel/idaFoodWeb_kry_p: examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/build.make
examples/ida/parallel/idaFoodWeb_kry_p: src/ida/libsundials_ida.so.2.9.0
examples/ida/parallel/idaFoodWeb_kry_p: src/nvec_par/libsundials_nvecparallel.so.2.7.0
examples/ida/parallel/idaFoodWeb_kry_p: /usr/lib/x86_64-linux-gnu/librt.so
examples/ida/parallel/idaFoodWeb_kry_p: examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable idaFoodWeb_kry_p"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/idaFoodWeb_kry_p.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/build: examples/ida/parallel/idaFoodWeb_kry_p

.PHONY : examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/build

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/requires: examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/idaFoodWeb_kry_p.c.o.requires

.PHONY : examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/requires

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel && $(CMAKE_COMMAND) -P CMakeFiles/idaFoodWeb_kry_p.dir/cmake_clean.cmake
.PHONY : examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/clean

examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/ida/parallel /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/ida/parallel/CMakeFiles/idaFoodWeb_kry_p.dir/depend

