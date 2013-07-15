find_package(PkgConfig)
pkg_check_modules(FFMPEG libavcodec>=54 libavformat>=54 libavutil>=51 libswscale>=2.1.100 REQUIRED)
include_directories(${FFMPEG_INCLUDE_DIRS})

add_library(avmm avmm.cc)
target_link_libraries(avmm ${FFMPEG_LIBRARIES})
