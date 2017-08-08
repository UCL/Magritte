# CMakeLists.txt for 3D-RT setup
# ------------------------------



# Specify the minimum required CMake version

cmake_minimum_required(VERSION 2.8)


# Execute the pre-setup

add_custom_target( PRE_SETUP ALL
                   COMMAND ./pre_setup \${PARAMETERS_FILE} )

# Create setup library

add_library( _setup   setup_tools.cpp ../src/setup_data_structures.cpp ../src/data_tools.cpp )


# Create executable for setup

add_executable( setup   setup.cpp )


# Link the setup library to the executable

target_link_libraries( setup   _setup )

