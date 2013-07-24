include(ResourceToCXXSource)

## LIBRARY TARGETS

# All library sources
set(CLDTCWT_SOURCES
    DTCWT/dtcwt.cc
    DTCWT/intDtcwt.cc
    DisplayOutput/Abs/abs.cc
    DisplayOutput/AbsToRGBA/absToRGBA.cc
    DisplayOutput/GreyscaleToRGBA/greyscaleToRGBA.cc
    DisplayOutput/calculator.cc
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
    KeypointDescriptor/extractDescriptors.cc
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
    MiscKernels/Rescale/rescale.cc
    hdf5/hdfwriter.cc
    util/clUtil.cc
    util/clUtilCV.cc
)

set(CLDTCWT_KERNEL_SOURCES
    DisplayOutput/AbsToRGBA/kernel.cl
    DisplayOutput/GreyscaleToRGBA/kernel.cl
    Filter/DecimateFilterX/kernel.cl
    Filter/DecimateFilterY/kernel.cl
    Filter/DecimateTripleFilterX/kernel.cl
    Filter/FilterX/kernel.cl
    Filter/FilterY/kernel.cl
    Filter/ImageToImageBuffer/kernel.cl
    Filter/PadX/kernel.cl
    Filter/PadY/kernel.cl
    Filter/QuadToComplex/kernel.cl
    Filter/QuadToComplexDecimateFilterY/kernel.cl
    Filter/ScaleImageToImageBuffer/kernel.cl
    Filter/TripleQuadToComplexDecimateFilterY/kernel.cl
    KeypointDescriptor/kernel.cl
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
)
resource_to_cxx_source(VARNAME CLDTCWT_COMPILED_KERNELS SOURCES ${CLDTCWT_KERNEL_SOURCES})

# A combined *shared* library for the entire system
add_library(cldtcwt SHARED
    ${CLDTCWT_SOURCES} ${CLDTCWT_COMPILED_KERNELS}
)
target_link_libraries(cldtcwt
    ${OPENCV_LIBRARIES}
    ${OPENCL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${HDF5_LIBRARIES}
)

# Set version and SOVERSION on library
set_target_properties(cldtcwt PROPERTIES
    SOVERSION ${CLDTCWT_MAJOR_VERSION}
    VERSION ${CLDTCWT_VERSION}
)

# Install binaries
install(TARGETS cldtcwt
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install header files. We simply glob for any file called *.{h,hpp} in the source
# directory. Something of a sledgehammer but it gets the job done(!)
install(
    DIRECTORY .
    DESTINATION include/cldtcwt
    FILES_MATCHING PATTERN *.h PATTERN *.hpp
)

# Create and install the pkg-config file
configure_file(cldtcwt.pc.in cldtcwt.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cldtcwt.pc DESTINATION lib/pkgconfig)
