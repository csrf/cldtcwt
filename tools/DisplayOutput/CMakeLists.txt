## DEPENDENCIES
#

find_package(PkgConfig REQUIRED)

pkg_check_modules(SFML sfml-system>=2.0 sfml-window>=2.0)
include_directories(${SFML_INCLUDE_DIRS})

pkg_check_modules(V4L2 libv4l2)
include_directories(${V4L2_INCLUDE_DIRS})

## EXECUTABLE TARGETS
#

if(SFML_FOUND)
    # The displayVideoDTCWT executable:

    add_executable(displayVideoDTCWT displayVideoDTCWT.cc)
    target_link_libraries(displayVideoDTCWT
        cldtcwt
        ${SFML_LIBRARIES}
    )

    install(
        TARGETS displayVideoDTCWT
        RUNTIME DESTINATION bin
    )
endif(SFML_FOUND)

if(SFML_FOUND AND V4L2_FOUND)
    # The displayDTCWT executable:

    add_executable(displayDTCWT displayDTCWT.cc)
    target_link_libraries(displayDTCWT
        cldtcwt
        ${SFML_LIBRARIES}
        ${V4L2_LIBRARIES}
    )

    install(
        TARGETS displayDTCWT
        RUNTIME DESTINATION bin
    )
endif(SFML_FOUND AND V4L2_FOUND)
