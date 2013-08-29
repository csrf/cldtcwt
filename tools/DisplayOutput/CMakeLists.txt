## DEPENDENCIES
#

find_package(PkgConfig REQUIRED)

pkg_check_modules(SFML sfml-system<2.0 sfml-window<2.0)
if(NOT SFML_FOUND)
    # If pkg-config failed, fall back to searching directories
    find_package(SFML)
    set(SFML_INCLUDE_DIRS ${SFML_WINDOW_INCLUDE_DIR})
    set(SFML_LIBRARIES ${SFML_WINDOW_LIBRARY_SHARED_NONDEBUG})
endif(NOT SFML_FOUND)
include_directories(${SFML_INCLUDE_DIRS})

pkg_check_modules(FFMPEG libavcodec>=54 libavformat>=54 libavutil>=51 libswscale>=2.1.100)
include_directories(${FFMPEG_INCLUDE_DIRS})

pkg_check_modules(V4L2 libv4l2)
include_directories(${V4L2_INCLUDE_DIRS})

## EXECUTABLE TARGETS
#

if(SFML_FOUND)
    if(FFMPEG_FOUND)
        # The displayVideoDTCWT executable:
        add_executable(displayVideoDTCWT
            displayVideoDTCWT.cc
            avmm/avmm.cc

            VBOBuffer.cc
            calculatorInterface.cc
            texture.cc
            viewer.cc
        )
        target_link_libraries(displayVideoDTCWT
            cldtcwt
            ${SFML_LIBRARIES}
            ${FFMPEG_LIBRARIES}
        )

        install(
            TARGETS displayVideoDTCWT
            RUNTIME DESTINATION bin
        )
    endif(FFMPEG_FOUND)

    if(V4L2_FOUND)
        # The displayDTCWT executable:

        add_executable(displayDTCWT
            displayDTCWT.cc
            VideoReader.cc

            VBOBuffer.cc
            calculatorInterface.cc
            texture.cc
            viewer.cc
        )
        target_link_libraries(displayDTCWT
            cldtcwt
            ${SFML_LIBRARIES}
            ${V4L2_LIBRARIES}
        )

        install(
            TARGETS displayDTCWT
            RUNTIME DESTINATION bin
        )
    endif(V4L2_FOUND)
endif(SFML_FOUND)
