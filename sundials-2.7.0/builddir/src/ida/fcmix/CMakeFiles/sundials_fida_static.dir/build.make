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
include src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/depend.make

# Include the progress variables for this target.
include src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/progress.make

# Include the compile flags for this target's objects.
include src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o: ../src/ida/fcmix/fidaband.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fidaband.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaband.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fidaband.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaband.c > CMakeFiles/sundials_fida_static.dir/fidaband.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fidaband.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaband.c -o CMakeFiles/sundials_fida_static.dir/fidaband.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o


src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o: ../src/ida/fcmix/fidabbd.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fidabbd.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidabbd.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fidabbd.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidabbd.c > CMakeFiles/sundials_fida_static.dir/fidabbd.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fidabbd.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidabbd.c -o CMakeFiles/sundials_fida_static.dir/fidabbd.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o


src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o: ../src/ida/fcmix/fida.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fida.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fida.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fida.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fida.c > CMakeFiles/sundials_fida_static.dir/fida.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fida.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fida.c -o CMakeFiles/sundials_fida_static.dir/fida.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o


src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o: ../src/ida/fcmix/fidadense.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fidadense.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidadense.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fidadense.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidadense.c > CMakeFiles/sundials_fida_static.dir/fidadense.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fidadense.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidadense.c -o CMakeFiles/sundials_fida_static.dir/fidadense.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o


src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o: ../src/ida/fcmix/fidaewt.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fidaewt.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaewt.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fidaewt.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaewt.c > CMakeFiles/sundials_fida_static.dir/fidaewt.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fidaewt.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaewt.c -o CMakeFiles/sundials_fida_static.dir/fidaewt.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o


src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o: ../src/ida/fcmix/fidajtimes.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidajtimes.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fidajtimes.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidajtimes.c > CMakeFiles/sundials_fida_static.dir/fidajtimes.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fidajtimes.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidajtimes.c -o CMakeFiles/sundials_fida_static.dir/fidajtimes.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o


src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o: ../src/ida/fcmix/fidapreco.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fidapreco.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidapreco.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fidapreco.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidapreco.c > CMakeFiles/sundials_fida_static.dir/fidapreco.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fidapreco.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidapreco.c -o CMakeFiles/sundials_fida_static.dir/fidapreco.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o


src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/flags.make
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o: ../src/ida/fcmix/fidaroot.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sundials_fida_static.dir/fidaroot.c.o   -c /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaroot.c

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sundials_fida_static.dir/fidaroot.c.i"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaroot.c > CMakeFiles/sundials_fida_static.dir/fidaroot.c.i

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sundials_fida_static.dir/fidaroot.c.s"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix/fidaroot.c -o CMakeFiles/sundials_fida_static.dir/fidaroot.c.s

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.requires:

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.provides: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.requires
	$(MAKE) -f src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.provides.build
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.provides

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.provides.build: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o


# Object files for target sundials_fida_static
sundials_fida_static_OBJECTS = \
"CMakeFiles/sundials_fida_static.dir/fidaband.c.o" \
"CMakeFiles/sundials_fida_static.dir/fidabbd.c.o" \
"CMakeFiles/sundials_fida_static.dir/fida.c.o" \
"CMakeFiles/sundials_fida_static.dir/fidadense.c.o" \
"CMakeFiles/sundials_fida_static.dir/fidaewt.c.o" \
"CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o" \
"CMakeFiles/sundials_fida_static.dir/fidapreco.c.o" \
"CMakeFiles/sundials_fida_static.dir/fidaroot.c.o"

# External object files for target sundials_fida_static
sundials_fida_static_EXTERNAL_OBJECTS =

src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build.make
src/ida/fcmix/libsundials_fida.a: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking C static library libsundials_fida.a"
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && $(CMAKE_COMMAND) -P CMakeFiles/sundials_fida_static.dir/cmake_clean_target.cmake
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sundials_fida_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build: src/ida/fcmix/libsundials_fida.a

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/build

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaband.c.o.requires
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidabbd.c.o.requires
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fida.c.o.requires
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidadense.c.o.requires
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaewt.c.o.requires
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidajtimes.c.o.requires
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidapreco.c.o.requires
src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires: src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/fidaroot.c.o.requires

.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/requires

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/clean:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix && $(CMAKE_COMMAND) -P CMakeFiles/sundials_fida_static.dir/cmake_clean.cmake
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/clean

src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/depend:
	cd /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0 /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/src/ida/fcmix /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix /home/frederik/Dropbox/Astro/3D-RT/sundials-2.7.0/builddir/src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/ida/fcmix/CMakeFiles/sundials_fida_static.dir/depend

