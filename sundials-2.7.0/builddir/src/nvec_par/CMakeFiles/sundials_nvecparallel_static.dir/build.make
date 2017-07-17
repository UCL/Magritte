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
include src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/depend.make

# Include the progress variables for this target.
include src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/progress.make

# Include the compile flags for this target's objects.
include src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/flags.make

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/flags.make
src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o: ../src/nvec_par/nvector_parallel.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/nvec_par/nvector_parallel.c

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/nvec_par/nvector_parallel.c > CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.i

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/nvec_par/nvector_parallel.c -o CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.s

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.requires:

.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.requires

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.provides: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.requires
	$(MAKE) -f src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/build.make src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.provides.build
.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.provides

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.provides.build: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o


src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/flags.make
src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o: ../src/sundials/sundials_math.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/sundials/sundials_math.c

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/sundials/sundials_math.c > CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.i

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && /opt/intel/compilers_and_libraries_2017.2.174/linux/mpi/intel64/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/sundials/sundials_math.c -o CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.s

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.requires:

.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.requires

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.provides: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.requires
	$(MAKE) -f src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/build.make src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.provides.build
.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.provides

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.provides.build: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o


# Object files for target sundials_nvecparallel_static
sundials_nvecparallel_static_OBJECTS = \
"CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o" \
"CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o"

# External object files for target sundials_nvecparallel_static
sundials_nvecparallel_static_EXTERNAL_OBJECTS =

src/nvec_par/libsundials_nvecparallel.a: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o
src/nvec_par/libsundials_nvecparallel.a: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o
src/nvec_par/libsundials_nvecparallel.a: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/build.make
src/nvec_par/libsundials_nvecparallel.a: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C static library libsundials_nvecparallel.a"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && $(CMAKE_COMMAND) -P CMakeFiles/sundials_nvecparallel_static.dir/cmake_clean_target.cmake
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sundials_nvecparallel_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/build: src/nvec_par/libsundials_nvecparallel.a

.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/build

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/requires: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/nvector_parallel.c.o.requires
src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/requires: src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/__/sundials/sundials_math.c.o.requires

.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/requires

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par && $(CMAKE_COMMAND) -P CMakeFiles/sundials_nvecparallel_static.dir/cmake_clean.cmake
.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/clean

src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/nvec_par /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/nvec_par/CMakeFiles/sundials_nvecparallel_static.dir/depend

