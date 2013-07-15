add_library(hdf5 hdfwriter.cc)
target_link_libraries(hdf5 ${HDF5_LIBRARIES})

add_executable(hdf5_test test.cc)
target_link_libraries(hdf5_test hdf5)
add_test(HDF5 hdf5_test)
