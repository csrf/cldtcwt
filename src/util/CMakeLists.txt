add_library(clutil
    clUtil.cc
    clUtil.h
    clUtilCV.cc
    clUtilCV.h
)
target_link_libraries(clutil ${OPENCL_LIBRARIES} ${OPENCV_LIBRARIES})
