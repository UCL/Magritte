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
CMAKE_SOURCE_DIR = /home/frederik/Dropbox/Astro/3D-RT/setup

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frederik/Dropbox/Astro/3D-RT/setup

# Utility rule file for PRE_SETUP.

# Include the progress variables for this target.
include CMakeFiles/PRE_SETUP.dir/progress.make

CMakeFiles/PRE_SETUP:
	./pre_setup ${PARAMETERS_FILE}

PRE_SETUP: CMakeFiles/PRE_SETUP
PRE_SETUP: CMakeFiles/PRE_SETUP.dir/build.make

.PHONY : PRE_SETUP

# Rule to build all files generated by this target.
CMakeFiles/PRE_SETUP.dir/build: PRE_SETUP

.PHONY : CMakeFiles/PRE_SETUP.dir/build

CMakeFiles/PRE_SETUP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PRE_SETUP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PRE_SETUP.dir/clean

CMakeFiles/PRE_SETUP.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/setup && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup/CMakeFiles/PRE_SETUP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PRE_SETUP.dir/depend

