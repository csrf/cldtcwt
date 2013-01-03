// PADDING should have been defined externally, as should WG_W
// and WG_H (width and height of the workgroup respectively)
__attribute__((reqd_work_group_size(WG_W, WG_H, 1)))
__kernel void imageToImageBuffer(__read_only image2d_t input,
                                 unsigned int stride,
                                 __global float* output)
                            
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST;

    const int2 g = (int2) (get_global_id(0), get_global_id(1));

    // Copy to the image buffer
    output[(PADDING + g.y) * stride
           + PADDING + g.x]
        = read_imagef(input, sampler, g);

}


