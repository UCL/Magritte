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

# Include any dependencies generated for this target.
include CMakeFiles/_setup.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/_setup.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/_setup.dir/flags.make

CMakeFiles/_setup.dir/setup_tools.cpp.o: CMakeFiles/_setup.dir/flags.make
CMakeFiles/_setup.dir/setup_tools.cpp.o: setup_tools.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/setup/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/_setup.dir/setup_tools.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_setup.dir/setup_tools.cpp.o -c /home/frederik/Dropbox/Astro/3D-RT/setup/setup_tools.cpp

CMakeFiles/_setup.dir/setup_tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_setup.dir/setup_tools.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/setup/setup_tools.cpp > CMakeFiles/_setup.dir/setup_tools.cpp.i

CMakeFiles/_setup.dir/setup_tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_setup.dir/setup_tools.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/setup/setup_tools.cpp -o CMakeFiles/_setup.dir/setup_tools.cpp.s

CMakeFiles/_setup.dir/setup_tools.cpp.o.requires:

.PHONY : CMakeFiles/_setup.dir/setup_tools.cpp.o.requires

CMakeFiles/_setup.dir/setup_tools.cpp.o.provides: CMakeFiles/_setup.dir/setup_tools.cpp.o.requires
	$(MAKE) -f CMakeFiles/_setup.dir/build.make CMakeFiles/_setup.dir/setup_tools.cpp.o.provides.build
.PHONY : CMakeFiles/_setup.dir/setup_tools.cpp.o.provides

CMakeFiles/_setup.dir/setup_tools.cpp.o.provides.build: CMakeFiles/_setup.dir/setup_tools.cpp.o


CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o: CMakeFiles/_setup.dir/flags.make
CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o: /home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/setup/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o -c /home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp > CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.i

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp -o CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.s

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.requires:

.PHONY : CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.requires

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.provides: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.requires
	$(MAKE) -f CMakeFiles/_setup.dir/build.make CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.provides.build
.PHONY : CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.provides

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.provides.build: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o


CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o: CMakeFiles/_setup.dir/flags.make
CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o: /home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/setup/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o -c /home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp > CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.i

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp -o CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.s

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.requires:

.PHONY : CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.requires

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.provides: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.requires
	$(MAKE) -f CMakeFiles/_setup.dir/build.make CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.provides.build
.PHONY : CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.provides

CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.provides.build: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o


# Object files for target _setup
_setup_OBJECTS = \
"CMakeFiles/_setup.dir/setup_tools.cpp.o" \
"CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o" \
"CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o"

# External object files for target _setup
_setup_EXTERNAL_OBJECTS =

lib_setup.a: CMakeFiles/_setup.dir/setup_tools.cpp.o
lib_setup.a: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o
lib_setup.a: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o
lib_setup.a: CMakeFiles/_setup.dir/build.make
lib_setup.a: CMakeFiles/_setup.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/setup/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library lib_setup.a"
	$(CMAKE_COMMAND) -P CMakeFiles/_setup.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_setup.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/_setup.dir/build: lib_setup.a

.PHONY : CMakeFiles/_setup.dir/build

CMakeFiles/_setup.dir/requires: CMakeFiles/_setup.dir/setup_tools.cpp.o.requires
CMakeFiles/_setup.dir/requires: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/setup_data_structures.cpp.o.requires
CMakeFiles/_setup.dir/requires: CMakeFiles/_setup.dir/home/frederik/Dropbox/Astro/3D-RT/src/data_tools.cpp.o.requires

.PHONY : CMakeFiles/_setup.dir/requires

CMakeFiles/_setup.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/_setup.dir/cmake_clean.cmake
.PHONY : CMakeFiles/_setup.dir/clean

CMakeFiles/_setup.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/setup && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup /home/frederik/Dropbox/Astro/3D-RT/setup/CMakeFiles/_setup.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/_setup.dir/depend

