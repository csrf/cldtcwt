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
