set(SOURCES
    DecimateFilterX/decimateFilterX.cc
    DecimateFilterY/decimateFilterY.cc
    DecimateTripleFilterX/decimateTripleFilterX.cc
    FilterX/filterX.cc
    FilterY/filterY.cc
    ImageToImageBuffer/imageToImageBuffer.cc
    PadX/padX.cc
    PadY/padY.cc
    QuadToComplex/quadToComplex.cc
    QuadToComplexDecimateFilterY/q2cDecimateFilterY.cc
    ScaleImageToImageBuffer/scaleImageToImageBuffer.cc
    TripleQuadToComplexDecimateFilterY/tripleQ2cDecimateFilterY.cc
    imageBuffer.cc
    referenceImplementation.cc
)

set(RESOURCES
    ScaleImageToImageBuffer/kernel.cl
    PadY/kernel.cl
    FilterY/kernel.cl
    QuadToComplexDecimateFilterY/kernel.cl
    DecimateTripleFilterX/kernel.cl
    PadX/kernel.cl
    FilterX/kernel.cl
    TripleQuadToComplexDecimateFilterY/kernel.cl
    DecimateFilterX/kernel.cl
    QuadToComplex/kernel.cl
    DecimateFilterY/kernel.cl
    ImageToImageBuffer/kernel.cl
)

include(ResourceToCXXSource)
resource_to_cxx_source(VARNAME COMPILED_RESOURCES SOURCES ${RESOURCES})

add_library(bufferfilter ${SOURCES} ${COMPILED_RESOURCES})
target_link_libraries(bufferfilter clutil)

set(TEST_SOURCES
    DecimateFilterX/speedTestDecimateFilterX.cc
    DecimateFilterX/testDecimateFilterX.cc
    DecimateFilterY/speedTestDecimateFilterY.cc
    DecimateFilterY/testDecimateFilterY.cc
    DecimateTripleFilterX/speedTest.cc
    DecimateTripleFilterX/test.cc
    FilterX/testFilterX.cc
    FilterY/testFilterY.cc
    QuadToComplex/speedTest.cc
    QuadToComplex/test.cc
    QuadToComplexDecimateFilterY/speedTest.cc
    QuadToComplexDecimateFilterY/test.cc
    TripleQuadToComplexDecimateFilterY/speedTest.cc
    TripleQuadToComplexDecimateFilterY/test.cc
    speedTest.cc
)

include(AddTestSources)
add_test_sources(
    PREFIX BufferFilter
    LINK_LIBRARIES cldtcwt
    SOURCES ${TEST_SOURCES}
)
