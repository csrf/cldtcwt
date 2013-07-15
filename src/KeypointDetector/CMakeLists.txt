set(SOURCES
    Accumulate/accumulate.cc
    Concat/concat.cc
    EnergyMaps/BTK/energyMap.cc
    EnergyMaps/CrossProduct/crossProduct.cc
    EnergyMaps/Eigen/energyMapEigen.cc
    EnergyMaps/EnergyMap/energyMap.cc
    EnergyMaps/InterpMap/interpMap.cc
    EnergyMaps/InterpPhaseMap/interpPhaseMap.cc
    EnergyMaps/PyramidSum/pyramidSum.cc
    FindMax/findMax.cc
    peakDetector.cc
)

set(RESOURCES
    Accumulate/kernel.cl
    Concat/kernel.cl
    EnergyMaps/BTK/kernel.cl
    EnergyMaps/CrossProduct/kernel.cl
    EnergyMaps/Eigen/kernel.cl
    EnergyMaps/EnergyMap/kernel.cl
    EnergyMaps/InterpMap/kernel.cl
    EnergyMaps/InterpPhaseMap/kernel.cl
    EnergyMaps/PyramidSum/kernel.cl
    FindMax/kernel.cl
)

include(ResourceToCXXSource)
resource_to_cxx_source(VARNAME COMPILED_RESOURCES SOURCES ${RESOURCES})

add_library(keypoint_detector ${SOURCES} ${COMPILED_RESOURCES})
target_link_libraries(keypoint_detector clutil)
