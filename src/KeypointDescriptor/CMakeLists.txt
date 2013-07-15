set(SOURCES
    extractDescriptors.cc
)

set(RESOURCES
    kernel.cl
)

include(ResourceToCXXSource)
resource_to_cxx_source(VARNAME COMPILED_RESOURCES SOURCES ${RESOURCES})

add_library(keypoint_descriptor ${SOURCES} ${COMPILED_RESOURCES})
target_link_libraries(keypoint_descriptor clutil)
