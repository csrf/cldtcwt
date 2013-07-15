include(CMakeParseArguments)

function(add_test_sources)
    set(ATS_PREFIX "Test")
    set(oneValueArgs PREFIX)
    set(multiValueArgs SOURCES LINK_LIBRARIES)
    cmake_parse_arguments(ATS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(_test_source ${ATS_SOURCES})
        get_filename_component(_path "${_test_source}" PATH)
        string(REPLACE "/" "" _mangled_path "${_path}")

        get_filename_component(_name "${_test_source}" NAME)
        get_filename_component(_name_we "${_test_source}" NAME_WE)
        set(_test_executable "test_${_mangled_path}_${_name_we}")

        add_executable(${_test_executable} ${_test_source})
        target_link_libraries(${_test_executable} ${ATS_LINK_LIBRARIES})
        add_test("${ATS_PREFIX}${_mangled_path}${_name_we}" ${_test_executable})
    endforeach()
endfunction(add_test_sources)
