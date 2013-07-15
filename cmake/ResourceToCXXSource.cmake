# Function to turn an OpenCL source file into a C string within a source file.
# xxd uses its input's filename to name the string and its length, so we
# need to move them to a name that depends only on the path output, not its
# input.  Otherwise, builds in different relative locations would put the
# source into different variable names, and everything would fall over.
# The actual name will be filename (.s replaced with underscores), and length
# name_len.
#
# Usage example:
#
# set(KERNELS a.cl b/c.cl)
# resource_to_cxx_source(
#   SOURCES ${KERNELS}
#   VARNAME OUTPUTS
# )
# add_executable(foo ${OUTPUTS})
#
# The namespace they are placed in is taken from filename.namespace.
#
# For example, if the input file is kernel.cl, the two variables will be
#  unsigned char ns::kernel_cl[];
#  unsigned int ns::kernel_cl_len;   
#
# where ns is the contents of kernel.cl.namespace.

include(CMakeParseArguments)

get_filename_component(XXD_COMPILER xxd PROGRAM)

function(resource_to_cxx_source)
    cmake_parse_arguments(RTCS "" "VARNAME" "SOURCES" ${ARGN})

    set(_output_files "")
    foreach(_input_file ${RTCS_SOURCES})
        file(READ "${_input_file}.namespace" _namespace)
        string(STRIP "${_namespace}" _namespace)

        get_filename_component(_path "${_input_file}" PATH)
        get_filename_component(_name "${_input_file}" NAME)
        get_filename_component(_name_we "${_input_file}" NAME_WE)
        set(_output_path "${CMAKE_CURRENT_BINARY_DIR}/${_path}")
        set(_output_file "${_output_path}/${_name_we}.cc")
        add_custom_command(
            OUTPUT ${_output_file}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_output_path}"
            COMMAND ${CMAKE_COMMAND} -E echo "namespace ${_namespace} {" >>"${_output_file}"
            COMMAND ${XXD_COMPILER} -i "${_name}" >>"${_output_file}"
            COMMAND ${CMAKE_COMMAND} -E echo "}" >>"${_output_file}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_path}"
            COMMENT "Compiling ${_input_file} to C++ source"
        )
    list(APPEND _output_files "${_output_file}")
    endforeach()

    set("${RTCS_VARNAME}" ${_output_files} PARENT_SCOPE)
endfunction(resource_to_cxx_source)
