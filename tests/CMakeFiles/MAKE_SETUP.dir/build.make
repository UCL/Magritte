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
CMAKE_SOURCE_DIR = /home/frederik/Dropbox/Astro/3D-RT/tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frederik/Dropbox/Astro/3D-RT/tests

# Utility rule file for MAKE_SETUP.

# Include the progress variables for this target.
include CMakeFiles/MAKE_SETUP.dir/progress.make

CMakeFiles/MAKE_SETUP:
	cd /home/frederik/Dropbox/Astro/3D-RT/setup && make PARAMETERS_FILE=../tests/parameters_for_testing.txt && ./setup

MAKE_SETUP: CMakeFiles/MAKE_SETUP
MAKE_SETUP: CMakeFiles/MAKE_SETUP.dir/build.make

.PHONY : MAKE_SETUP

# Rule to build all files generated by this target.
CMakeFiles/MAKE_SETUP.dir/build: MAKE_SETUP

.PHONY : CMakeFiles/MAKE_SETUP.dir/build

CMakeFiles/MAKE_SETUP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MAKE_SETUP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MAKE_SETUP.dir/clean

CMakeFiles/MAKE_SETUP.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/tests /home/frederik/Dropbox/Astro/3D-RT/tests /home/frederik/Dropbox/Astro/3D-RT/tests /home/frederik/Dropbox/Astro/3D-RT/tests /home/frederik/Dropbox/Astro/3D-RT/tests/CMakeFiles/MAKE_SETUP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MAKE_SETUP.dir/depend

