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

# Include any dependencies generated for this target.
include test_src/CMakeFiles/_test_main.dir/depend.make

# Include the progress variables for this target.
include test_src/CMakeFiles/_test_main.dir/progress.make

# Include the compile flags for this target's objects.
include test_src/CMakeFiles/_test_main.dir/flags.make

test_src/CMakeFiles/_test_main.dir/test_main.cpp.o: test_src/CMakeFiles/_test_main.dir/flags.make
test_src/CMakeFiles/_test_main.dir/test_main.cpp.o: test_src/test_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test_src/CMakeFiles/_test_main.dir/test_main.cpp.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/tests/test_src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_test_main.dir/test_main.cpp.o -c /home/frederik/Dropbox/Astro/3D-RT/tests/test_src/test_main.cpp

test_src/CMakeFiles/_test_main.dir/test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_test_main.dir/test_main.cpp.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/tests/test_src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/tests/test_src/test_main.cpp > CMakeFiles/_test_main.dir/test_main.cpp.i

test_src/CMakeFiles/_test_main.dir/test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_test_main.dir/test_main.cpp.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/tests/test_src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/tests/test_src/test_main.cpp -o CMakeFiles/_test_main.dir/test_main.cpp.s

test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.requires:

.PHONY : test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.requires

test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.provides: test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.requires
	$(MAKE) -f test_src/CMakeFiles/_test_main.dir/build.make test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.provides.build
.PHONY : test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.provides

test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.provides.build: test_src/CMakeFiles/_test_main.dir/test_main.cpp.o


# Object files for target _test_main
_test_main_OBJECTS = \
"CMakeFiles/_test_main.dir/test_main.cpp.o"

# External object files for target _test_main
_test_main_EXTERNAL_OBJECTS =

test_src/lib_test_main.a: test_src/CMakeFiles/_test_main.dir/test_main.cpp.o
test_src/lib_test_main.a: test_src/CMakeFiles/_test_main.dir/build.make
test_src/lib_test_main.a: test_src/CMakeFiles/_test_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library lib_test_main.a"
	cd /home/frederik/Dropbox/Astro/3D-RT/tests/test_src && $(CMAKE_COMMAND) -P CMakeFiles/_test_main.dir/cmake_clean_target.cmake
	cd /home/frederik/Dropbox/Astro/3D-RT/tests/test_src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_test_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_src/CMakeFiles/_test_main.dir/build: test_src/lib_test_main.a

.PHONY : test_src/CMakeFiles/_test_main.dir/build

test_src/CMakeFiles/_test_main.dir/requires: test_src/CMakeFiles/_test_main.dir/test_main.cpp.o.requires

.PHONY : test_src/CMakeFiles/_test_main.dir/requires

test_src/CMakeFiles/_test_main.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/tests/test_src && $(CMAKE_COMMAND) -P CMakeFiles/_test_main.dir/cmake_clean.cmake
.PHONY : test_src/CMakeFiles/_test_main.dir/clean

test_src/CMakeFiles/_test_main.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/tests /home/frederik/Dropbox/Astro/3D-RT/tests/test_src /home/frederik/Dropbox/Astro/3D-RT/tests /home/frederik/Dropbox/Astro/3D-RT/tests/test_src /home/frederik/Dropbox/Astro/3D-RT/tests/test_src/CMakeFiles/_test_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test_src/CMakeFiles/_test_main.dir/depend
