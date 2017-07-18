# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build

# Include any dependencies generated for this target.
include examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/depend.make

# Include the progress variables for this target.
include examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/progress.make

# Include the compile flags for this target's objects.
include examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/flags.make

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o: examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/flags.make
examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o: /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/C_openmp/cvAdvDiff_bnd_omp.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp && /opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/C_openmp/cvAdvDiff_bnd_omp.c

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp && /opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/C_openmp/cvAdvDiff_bnd_omp.c > CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.i

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp && /opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/C_openmp/cvAdvDiff_bnd_omp.c -o CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.s

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.requires:

.PHONY : examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.requires

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.provides: examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.requires
	$(MAKE) -f examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/build.make examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.provides.build
.PHONY : examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.provides

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.provides.build: examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o


# Object files for target cvAdvDiff_bnd_omp
cvAdvDiff_bnd_omp_OBJECTS = \
"CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o"

# External object files for target cvAdvDiff_bnd_omp
cvAdvDiff_bnd_omp_EXTERNAL_OBJECTS =

examples/cvode/C_openmp/cvAdvDiff_bnd_omp: examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o
examples/cvode/C_openmp/cvAdvDiff_bnd_omp: examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/build.make
examples/cvode/C_openmp/cvAdvDiff_bnd_omp: src/cvode/libsundials_cvode.so.2.9.0
examples/cvode/C_openmp/cvAdvDiff_bnd_omp: src/nvec_openmp/libsundials_nvecopenmp.so.2.7.0
examples/cvode/C_openmp/cvAdvDiff_bnd_omp: /usr/lib/x86_64-linux-gnu/librt.so
examples/cvode/C_openmp/cvAdvDiff_bnd_omp: examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable cvAdvDiff_bnd_omp"
	cd /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvAdvDiff_bnd_omp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/build: examples/cvode/C_openmp/cvAdvDiff_bnd_omp

.PHONY : examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/build

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/requires: examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/cvAdvDiff_bnd_omp.c.o.requires

.PHONY : examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/requires

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp && $(CMAKE_COMMAND) -P CMakeFiles/cvAdvDiff_bnd_omp.dir/cmake_clean.cmake
.PHONY : examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/clean

examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/cvode/C_openmp /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/build/examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/cvode/C_openmp/CMakeFiles/cvAdvDiff_bnd_omp.dir/depend

