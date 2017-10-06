# CMakeLists.txt for Magritte setup
# ------------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Add Python libraries for make_rates.py

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


# Execute the pre-setup

add_custom_target( PRE_SETUP ALL
                   COMMAND ./pre_setup.exe \${PARAMETERS_FILE} )

# Create setup library

add_library( _setup   setup_tools.cpp ../src/setup_data_structures.cpp ../src/initializers.cpp ../src/data_tools.cpp )


# Create executable for setup

add_executable( setup.exe   setup.cpp )


# Link the setup library to the executable

target_link_libraries( setup.exe   _setup   ${PYTHON_LIBRARIES} )

