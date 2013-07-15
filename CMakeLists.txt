# CMake project boilerplate
project(clDTCWT CXX C)
cmake_minimum_required(VERSION 2.8)
enable_testing()

# Add the CMake modules we ship to the CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Use GNU's C++11 dialect
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

## DEPENDENCIES
#
# Only put global dependencies at this point. That is to say dependencies which
# are needed by more than one target. Single-target only dependencies are found
# in the corresponding CMakeLists file for that target.

find_package(PkgConfig REQUIRED)

pkg_check_modules(EIGEN eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

pkg_check_modules(OPENCV opencv REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INDLUDE_DIRS})

## TARGETS

# Add subdirectory containing the bulk of the source to be compiled.
add_subdirectory(src)

## DOCUMENTATION
#
# Create a doc target which will build doxygen documentation for this project.

find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)
