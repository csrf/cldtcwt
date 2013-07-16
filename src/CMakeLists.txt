# This directory is the #include root
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

## LIBRARY TARGETS

# All library sources
set(CLDTCWT_SOURCES
    DTCWT/dtcwt.cc
    DTCWT/intDtcwt.cc

    Filter/DecimateFilterX/decimateFilterX.cc
    Filter/DecimateFilterY/decimateFilterY.cc
    Filter/DecimateTripleFilterX/decimateTripleFilterX.cc
    Filter/FilterX/filterX.cc
    Filter/FilterY/filterY.cc
    Filter/ImageToImageBuffer/imageToImageBuffer.cc
    Filter/PadX/padX.cc
    Filter/PadY/padY.cc
    Filter/QuadToComplex/quadToComplex.cc
    Filter/QuadToComplexDecimateFilterY/q2cDecimateFilterY.cc
    Filter/ScaleImageToImageBuffer/scaleImageToImageBuffer.cc
    Filter/TripleQuadToComplexDecimateFilterY/tripleQ2cDecimateFilterY.cc
    Filter/imageBuffer.cc
    Filter/referenceImplementation.cc

    KeypointDetector/Accumulate/accumulate.cc
    KeypointDetector/Concat/concat.cc
    KeypointDetector/EnergyMaps/BTK/energyMap.cc
    KeypointDetector/EnergyMaps/CrossProduct/crossProduct.cc
    KeypointDetector/EnergyMaps/Eigen/energyMapEigen.cc
    KeypointDetector/EnergyMaps/EnergyMap/energyMap.cc
    KeypointDetector/EnergyMaps/InterpMap/interpMap.cc
    KeypointDetector/EnergyMaps/InterpPhaseMap/interpPhaseMap.cc
    KeypointDetector/EnergyMaps/PyramidSum/pyramidSum.cc
    KeypointDetector/FindMax/findMax.cc
    KeypointDetector/peakDetector.cc

    KeypointDescriptor/extractDescriptors.cc

    MiscKernels/Rescale/rescale.cc

    avmm/avmm.cc

    hdf5/hdfwriter.cc

    util/clUtil.cc
    util/clUtilCV.cc
)

set(CLDTCWT_KERNEL_SOURCES
    Filter/ScaleImageToImageBuffer/kernel.cl
    Filter/PadY/kernel.cl
    Filter/FilterY/kernel.cl
    Filter/QuadToComplexDecimateFilterY/kernel.cl
    Filter/DecimateTripleFilterX/kernel.cl
    Filter/PadX/kernel.cl
    Filter/FilterX/kernel.cl
    Filter/TripleQuadToComplexDecimateFilterY/kernel.cl
    Filter/DecimateFilterX/kernel.cl
    Filter/QuadToComplex/kernel.cl
    Filter/DecimateFilterY/kernel.cl
    Filter/ImageToImageBuffer/kernel.cl

    KeypointDetector/Accumulate/kernel.cl
    KeypointDetector/Concat/kernel.cl
    KeypointDetector/EnergyMaps/BTK/kernel.cl
    KeypointDetector/EnergyMaps/CrossProduct/kernel.cl
    KeypointDetector/EnergyMaps/Eigen/kernel.cl
    KeypointDetector/EnergyMaps/EnergyMap/kernel.cl
    KeypointDetector/EnergyMaps/InterpMap/kernel.cl
    KeypointDetector/EnergyMaps/InterpPhaseMap/kernel.cl
    KeypointDetector/EnergyMaps/PyramidSum/kernel.cl
    KeypointDetector/FindMax/kernel.cl

    KeypointDescriptor/kernel.cl
)

include(ResourceToCXXSource)
resource_to_cxx_source(VARNAME CLDTCWT_COMPILED_KERNELS SOURCES ${CLDTCWT_KERNEL_SOURCES})

# A combined *shared* library for the entire system
add_library(cldtcwt SHARED
    ${CLDTCWT_SOURCES} ${CLDTCWT_COMPILED_KERNELS}
)
target_link_libraries(cldtcwt
    ${OPENCV_LIBRARIES}
    ${OPENCL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${SFML_LIBRARIES}
    ${FFMPEG_LIBRARIES}
    ${HDF5_LIBRARIES}
)

# Install binaries
install(TARGETS cldtcwt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install header files. We simply glob for any file called *.h in the source
# directory. Something of a sledgehammer but it gets the job done(!)
install(
    DIRECTORY .
    DESTINATION include/cldtcwt
    FILES_MATCHING PATTERN *.h
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

    Filter/DecimateFilterX/speedTestDecimateFilterX.cc
    Filter/DecimateFilterX/testDecimateFilterX.cc
    Filter/DecimateFilterY/speedTestDecimateFilterY.cc
    Filter/DecimateFilterY/testDecimateFilterY.cc
    Filter/DecimateTripleFilterX/speedTest.cc
    Filter/DecimateTripleFilterX/test.cc
    Filter/FilterX/testFilterX.cc
    Filter/FilterY/testFilterY.cc
    Filter/QuadToComplex/speedTest.cc
    Filter/QuadToComplex/test.cc
    Filter/QuadToComplexDecimateFilterY/speedTest.cc
    Filter/QuadToComplexDecimateFilterY/test.cc
    Filter/TripleQuadToComplexDecimateFilterY/speedTest.cc
    Filter/TripleQuadToComplexDecimateFilterY/test.cc
    Filter/speedTest.cc
)

include(AddTestSources)
add_test_sources(
    LINK_LIBRARIES cldtcwt
    SOURCES ${TEST_SOURCES}
)

add_executable(hdf5_test hdf5/test.cc)
target_link_libraries(hdf5_test cldtcwt)
add_test(HDF5 hdf5_test)
