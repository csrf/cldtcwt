#ifndef HDFWRITER_H
#define HDFWRITER_H


#include <H5Cpp.h>

class HDFWriter {
    // Class to create and write to an HDF file
    //
    // Clears any existing contents and creates three tables:
    //   frames
    //      One row per frame, listing the starting keypoint (for the other two
    //      tables) and the number of keypoints.
    //   keypoints
    //      One row per keypoint.  x, y, scale, keypoint strength.
    //   descriptors
    //      One row per descriptor (lines up with the appropriate keypoint).
    //

    H5::H5File file;

    H5::DataSet frames, keypoints, descriptors;


public:

    HDFWriter() = default;
    HDFWriter(const HDFWriter&) = default;

    HDFWriter(std::string filename, size_t descriptorLength);

    void append(size_t numKeypoints, const float* keypoints,
                const float* descriptors);
    //   Appends adds an extra frame with associated keypoints

};



#endif

