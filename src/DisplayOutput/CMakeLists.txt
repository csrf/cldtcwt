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

set(DISPLAY_VIDEO_SOURCES
    Abs/abs.cc 
    AbsToRGBA/absToRGBA.cc 
    GreyscaleToRGBA/greyscaleToRGBA.cc 
    VBOBuffer.cc 
    calculator.cc 
    calculatorInterface.cc 
    displayVideoDTCWT.cc 
    texture.cc 
    viewer.cc
)

set(DISPLAY_VIDEO_RESOURCES
    AbsToRGBA/kernel.cl
    GreyscaleToRGBA/kernel.cl
)

include(ResourceToCXXSource)
resource_to_cxx_source(
    VARNAME COMPILED_DISPLAY_VIDEO_RESOURCES
    SOURCES ${DISPLAY_VIDEO_RESOURCES}
)

add_executable(displayVideoDTCWT
    ${DISPLAY_VIDEO_SOURCES}
    ${COMPILED_DISPLAY_VIDEO_RESOURCES}
)
target_link_libraries(displayVideoDTCWT
    cldtcwt
    ${SFML_LIBRARIES}
)

# The displayDTCWT executable:

set(DISPLAY_SOURCES
    Abs/abs.cc 
    AbsToRGBA/absToRGBA.cc 
    GreyscaleToRGBA/greyscaleToRGBA.cc 
    VBOBuffer.cc 
    VideoReader.cc
    calculator.cc 
    calculatorInterface.cc 
    displayDTCWT.cc 
    texture.cc 
    viewer.cc
)

set(DISPLAY_RESOURCES
    AbsToRGBA/kernel.cl
    GreyscaleToRGBA/kernel.cl
)

include(ResourceToCXXSource)
resource_to_cxx_source(
    VARNAME COMPILED_DISPLAY_RESOURCES
    SOURCES ${DISPLAY_RESOURCES}
)

add_executable(displayDTCWT
    ${DISPLAY_SOURCES}
    ${COMPILED_DISPLAY_RESOURCES}
)
target_link_libraries(displayDTCWT
    cldtcwt
    ${SFML_LIBRARIES}
    ${V4L2_LIBRARIES}
)

install(
    TARGETS displayVideoDTCWT displayDTCWT
    RUNTIME DESTINATION bin
)
