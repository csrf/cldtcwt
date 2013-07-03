#include "hdfwriter.h"


HDFWriter::HDFWriter(std::string filename, size_t descriptorLength)
{
    file = H5::H5File(H5std_string(filename.c_str()), H5F_ACC_TRUNC);

    // Create the frames dataset (pairs of starting index, number of
    // keypoints/descriptors)
    H5::DSetCreatPropList framesCparms;
    hsize_t chunkDims[] = {1024, 2};
    framesCparms.setChunk(2, chunkDims);

    const hsize_t framesDims[] = {0, 2};
    const hsize_t framesMaxDims[] = {H5S_UNLIMITED, 2};
    H5::DataSpace framesDataspace(2, framesDims, framesMaxDims);

    frames = file.createDataSet(H5std_string("frames"), 
                                H5::PredType::STD_U64LE,
                                framesDataspace,
                                framesCparms);

    // Create the keypoints dataset (x, y, scale, weight)
    H5::DSetCreatPropList keypointsCparms;
    chunkDims[1] = 4;
    keypointsCparms.setChunk(2, chunkDims);

    const hsize_t keypointsDims[] = {0, 4};
    const hsize_t keypointsMaxDims[] = {H5S_UNLIMITED, 4};
    H5::DataSpace keypointsDataspace(2, keypointsDims, keypointsMaxDims);

    keypoints = file.createDataSet(H5std_string("keypoints"), 
                                   H5::PredType::NATIVE_FLOAT,
                                   keypointsDataspace,
                                   keypointsCparms);

    // Create the descriptors dataset (descriptorLength elements)
    H5::DSetCreatPropList descriptorsCparms;
    chunkDims[1] = descriptorLength;
    descriptorsCparms.setChunk(2, chunkDims);

    const hsize_t descriptorsDims[] = {0, descriptorLength};
    const hsize_t descriptorsMaxDims[] = {H5S_UNLIMITED, descriptorLength};
    H5::DataSpace descriptorsDataspace(2, descriptorsDims, descriptorsMaxDims);

    descriptors = file.createDataSet(H5std_string("descriptors"), 
                                   H5::PredType::NATIVE_FLOAT,
                                   descriptorsDataspace,
                                   descriptorsCparms);
                                
}


// Add rows onto a 2D table
template<typename T>
void appendRows(H5::DataSet dataset, size_t numRows,
                const T* data, const H5::DataType& memType)
{    
    // Work out where we need to start writing from
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t existingDims[2];
    dataspace.getSimpleExtentDims(existingDims);

    hsize_t newDims[] = {existingDims[0] + numRows,
                         existingDims[1]};

    // Expand for the new data
    dataset.extend(newDims);
    dataspace = dataset.getSpace();


    // Select the new range
    hsize_t offset[] = {existingDims[0], 0};
    hsize_t range[] = {numRows, existingDims[1]};
    dataspace.selectHyperslab(H5S_SELECT_SET, range, offset);

    H5::DataSpace memSpace(2, range);

    // Copy across
    dataset.write(data, memType, memSpace, dataspace);
}


void HDFWriter::append(size_t numKeypoints,
                       const float *keypointsData,
                       const float *descriptorsData) 
{
    // Work out where the first keypoint will be
    hsize_t keypointDims[2];
    H5::DataSpace ds = keypoints.getSpace();
    ds.getSimpleExtentDims(keypointDims);

    hsize_t position[] = {keypointDims[0], numKeypoints};

    appendRows(frames, 1, position,
               H5::PredType::NATIVE_HSIZE);

    appendRows(keypoints, numKeypoints, keypointsData, 
               H5::PredType::NATIVE_FLOAT);

    appendRows(descriptors, numKeypoints, descriptorsData, 
               H5::PredType::NATIVE_FLOAT);
}



