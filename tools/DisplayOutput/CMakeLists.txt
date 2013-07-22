## DEPENDENCIES
#

find_package(PkgConfig REQUIRED)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

pkg_check_modules(SFML sfml-system>=2.0 sfml-window>=2.0 REQUIRED)
include_directories(${SFML_INCLUDE_DIRS})

pkg_check_modules(V4L2 libv4l2 REQUIRED)
include_directories(${V4L2_INCLUDE_DIRS})

## EXECUTABLE TARGETS
#

# The displayVideoDTCWT executable:

add_executable(displayVideoDTCWT displayVideoDTCWT.cc)
target_link_libraries(displayVideoDTCWT
    cldtcwt
    ${SFML_LIBRARIES}
)

# The displayDTCWT executable:

add_executable(displayDTCWT displayDTCWT.cc)
target_link_libraries(displayDTCWT
    cldtcwt
    ${SFML_LIBRARIES}
    ${V4L2_LIBRARIES}
)

install(
    TARGETS displayVideoDTCWT displayDTCWT
    RUNTIME DESTINATION bin
)
