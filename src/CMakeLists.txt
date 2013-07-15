# This directory is the #include root
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

## LIBRARY TARGETS

# Individual component libraries
add_subdirectory(DTCWT)
add_subdirectory(Filter)
add_subdirectory(KeypointDescriptor)
add_subdirectory(KeypointDetector)
add_subdirectory(MiscKernels)
add_subdirectory(avmm)
add_subdirectory(util)

# A combined *shared* library for the entire system
add_library(cldtcwt SHARED)
target_link_libraries(cldtcwt
    avmm
    bufferfilter
    clutil
    dtcwt
    keypoint_descriptor
    keypoint_detector
    rescale
)
install(TARGETS cldtcwt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

## EXECUTABLE TARGETS

# The display{,Video}DTCWT executables
add_subdirectory(DisplayOutput)

## TESTS

set(TEST_SOURCES
    test/test.cc
    test/testAccumulate.cc
    test/testConcat.cc
    test/testFindMax.cc
    test/testPeakDetector.cc
    test/testPyramidSum.cc
    test/testRescale.cc
)

include(AddTestSources)
add_test_sources(
    LINK_LIBRARIES cldtcwt
    SOURCES ${TEST_SOURCES}
)
