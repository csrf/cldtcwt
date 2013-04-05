// WG_W and WG_H should have been defined externally (width and height of 
// the workgroup respectively)
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
__kernel void scaleImageToImageBuffer(__read_only image2d_t input,
                                 __global float* output,
                                 unsigned int outputStart,
                                 unsigned int outputStride,
                                 float outputCentreX,
                                 float outputCentreY,
                                 float invScaleFactor)
                            
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR;

    const int2 g = (int2) (get_global_id(0), get_global_id(1));

    const float2 inputCentre = convert_float2(get_image_dim(input) - 1) / 2.f;
    const float2 outputCentre = (float2) (outputCentreX, outputCentreY);

    const float2 inPos =
        invScaleFactor * (convert_float2(g) - outputCentre) + inputCentre;

    // Copy to the image buffer
    output[outputStart + g.y * outputStride + g.x]
        = read_imagef(input, sampler, inPos).s0;

}


