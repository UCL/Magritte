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
include examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/depend.make

# Include the progress variables for this target.
include examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/progress.make

# Include the compile flags for this target's objects.
include examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/flags.make

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o: examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/flags.make
examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o: ../examples/idas/serial/idasRoberts_FSA_dns.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/idas/serial/idasRoberts_FSA_dns.c

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/idas/serial/idasRoberts_FSA_dns.c > CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.i

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/idas/serial/idasRoberts_FSA_dns.c -o CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.s

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.requires:

.PHONY : examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.requires

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.provides: examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.requires
	$(MAKE) -f examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/build.make examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.provides.build
.PHONY : examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.provides

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.provides.build: examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o


# Object files for target idasRoberts_FSA_dns
idasRoberts_FSA_dns_OBJECTS = \
"CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o"

# External object files for target idasRoberts_FSA_dns
idasRoberts_FSA_dns_EXTERNAL_OBJECTS =

examples/idas/serial/idasRoberts_FSA_dns: examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o
examples/idas/serial/idasRoberts_FSA_dns: examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/build.make
examples/idas/serial/idasRoberts_FSA_dns: src/idas/libsundials_idas.so.1.3.0
examples/idas/serial/idasRoberts_FSA_dns: src/nvec_ser/libsundials_nvecserial.so.2.7.0
examples/idas/serial/idasRoberts_FSA_dns: /usr/lib/x86_64-linux-gnu/librt.so
examples/idas/serial/idasRoberts_FSA_dns: examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable idasRoberts_FSA_dns"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/idasRoberts_FSA_dns.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/build: examples/idas/serial/idasRoberts_FSA_dns

.PHONY : examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/build

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/requires: examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/idasRoberts_FSA_dns.c.o.requires

.PHONY : examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/requires

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial && $(CMAKE_COMMAND) -P CMakeFiles/idasRoberts_FSA_dns.dir/cmake_clean.cmake
.PHONY : examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/clean

examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/examples/idas/serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/idas/serial/CMakeFiles/idasRoberts_FSA_dns.dir/depend

