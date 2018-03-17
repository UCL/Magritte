# Specify the minimum required CMake version
cmake_minimum_required(VERSION 2.8)

# Add VTK packages
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Add Python packages
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

# Add Boost packages
if (${PYTHON_VERSION_MAJOR} EQUAL 3)
    FIND_PACKAGE(Boost COMPONENTS python3)
else()
    FIND_PACKAGE(Boost COMPONENTS python)
endif()

# Include and link
include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR} ${PYTHON_LIBRARIES})

# Build and link the module
add_library(MagritteCppFunctions SHARED MagritteCppFunctions.cpp)
target_link_libraries(MagritteCppFunctions ${VTK_LIBRARIES})
target_link_libraries(MagritteCppFunctions ${Boost_LIBRARIES})
target_link_libraries(MagritteCppFunctions ${PYTHON_LIBRARIES})

# Tweaks the name of the library to match what Python expects
set_target_properties(MagritteCppFunctions PROPERTIES SUFFIX .so)
set_target_properties(MagritteCppFunctions PROPERTIES PREFIX "")
