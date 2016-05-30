/*
 * kernel_linear_blender
 * input0
 * input1
 * output
 * window, pos_x, pos_y, width, height
 */

#ifndef BLEND_UV
#define BLEND_UV 0
#endif

#define fixed_pixels 8

inline float8
y_coeffs (float coeff_start)
{
    float8 coeffs;

    coeffs.s0 = coeff_start;
    coeffs.s1 = coeff_start + 1.0f;
    coeffs.s2 = coeff_start + 2.0f;
    coeffs.s3 = coeff_start + 3.0f;
    coeffs.s4 = coeff_start + 4.0f;
    coeffs.s5 = coeff_start + 5.0f;
    coeffs.s6 = coeff_start + 6.0f;
    coeffs.s7 = coeff_start + 7.0f;
    return coeffs;
}

inline float8
uv_coeffs (float coeff_start)
{
    float8 coeffs;

    coeffs.s0 = coeffs.s1 = coeff_start;
    coeffs.s2 = coeffs.s3 = coeff_start + 2.0f;
    coeffs.s4 = coeffs.s5 = coeff_start + 4.0f;
    coeffs.s6 = coeffs.s7 = coeff_start + 6.0f;
    return coeffs;
}

__kernel void
kernel_linear_blender (
    __read_only image2d_t input0, __read_only image2d_t input1,
    __write_only image2d_t output,
    uint4 window, uint blend_width, uint2 input_width)
{
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const int g_x = get_global_id (0);
    const int g_y = get_global_id (1);
    int2 out_pos = (int2) (g_x, g_y);

    int merg_x_start = window.x / fixed_pixels;
    int merge_width = window.z / fixed_pixels;
    int merge_x_end = merg_x_start + merge_width;
    int input1_width = input_width.y;

    int2 in0_pos = (int2) (g_x, g_y);
    int2 in1_pos = (int2) (g_x - merg_x_start , g_y);

    if (g_x < merg_x_start) {
        uint4 data = read_imageui (input0, sampler, in0_pos);
        write_imageui (output, out_pos, data);
        return;
    }

    if (g_x > merge_x_end) {
        uint4 data = read_imageui (input1, sampler, in1_pos);
        write_imageui (output, out_pos, data);
        return;
    }
    float8 data0 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input0, sampler, in0_pos))));
    float8 data1 = convert_float8(as_uchar8(convert_ushort4(read_imageui(input1, sampler, in1_pos))));
    float8 out_data;

    float coeff_step = 1.0f / (float)(fixed_pixels) / merge_width;
    float coeff_start = convert_float(in1_pos.x) * fixed_pixels;
    float8 coeffs;

#if BLEND_UV
    coeffs = uv_coeffs (coeff_start);
#else
    coeffs = y_coeffs (coeff_start);
#endif
    coeffs *= coeff_step;

    out_data = data0 + (data1 - data0) * coeffs;

    write_imageui(output, out_pos, convert_uint4(as_ushort4(convert_uchar8(out_data))));
}