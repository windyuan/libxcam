/*
 * kernel_gauss_lap_pyramid.cl
 * input0
 * input1
 * output
 * window, pos_x, pos_y, width, height
 */

#ifndef PYRAMID_UV
#define PYRAMID_UV 0
#endif

#define fixed_pixels 8
#define GAUSS_V_R 2
#define GAUSS_H_R 1
#define COEFF_MID 3
#define Y_GAUSS_SCALE_OFFSSET (-0.25f)

#define zero8 (float8)(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)

__constant const float coeffs[7] = {0.0f, 0.1f, 0.25f, 0.3f, 0.25f, 0.1f, 0.0f};

/*
 * input: RGBA-CL_UNSIGNED_INT16
 * output_gauss: RGBA-CL_UNSIGNED_INT8
 * output_lap:RGBA-CL_UNSIGNED_INT16
 * each work-item calc 2 lines
 */
__kernel void
kernel_gauss_lap_transform (
    __read_only image2d_t input,
    __write_only image2d_t output_gauss, __write_only image2d_t output_lap)
{
    int g_x = get_global_id (0);
    int g_y = get_global_id (1) * 2;
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    int2 pos[2] = {(int2)(g_x, g_y), (int2)(g_x, g_y + 1) };
    float8 data[2];
    data[0] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, pos[0]))));
    data[1] = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, pos[1]))));
    float8 result_pre[2] = {zero8, zero8};
    float8 result_next[2] = {zero8, zero8};
    float8 result_cur[2];
    result_cur[0] = data[0] * coeffs[COEFF_MID] + data[1] * coeffs[COEFF_MID + 1];
    result_cur[1] = data[1] * coeffs[COEFF_MID] + data[0] * coeffs[COEFF_MID + 1];

    float8 tmp_data;
    int i_ver;

#pragma unroll
    for (i_ver = -GAUSS_V_R; i_ver <= GAUSS_V_R + 1; i_ver++) {
        int cur_g_y = g_y + i_ver;
        float coeff0 = coeffs[i_ver + COEFF_MID];
        float coeff1 = coeffs[i_ver + COEFF_MID - 1];
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(g_x - 1, cur_g_y)))));
        result_pre[0] += tmp_data * coeff0;
        result_pre[1] += tmp_data * coeff1;

        if (i_ver != 0 && i_ver != 1) {
            tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(g_x, cur_g_y)))));
            result_cur[0] += tmp_data * coeff0;
            result_cur[1] += tmp_data * coeff1;
        }
        tmp_data = convert_float8(as_uchar8(convert_ushort4(read_imageui(input, sampler, (int2)(g_x + 1, cur_g_y)))));
        result_next[0] += tmp_data * coeff0;
        result_next[1] += tmp_data * coeff1;
    }

    int i_line;
#pragma unroll
    for (i_line = 0; i_line < 2; ++i_line) {
        result_cur[i_line] = result_cur[i_line] * coeffs[COEFF_MID] +
                             (float8)(result_pre[i_line].s7, result_cur[i_line].s0123456) * coeffs[COEFF_MID + 1] +
                             (float8)(result_pre[i_line].s67, result_cur[i_line].s012345) * coeffs[COEFF_MID + 2] +
                             (float8)(result_cur[i_line].s1234567, result_next[i_line].s0) * coeffs[COEFF_MID + 1] +
                             (float8)(result_cur[i_line].s234567, result_next[i_line].s01) * coeffs[COEFF_MID + 2];

        tmp_data = data[i_line] - result_cur[i_line];
        tmp_data = tmp_data * 0.5f + 128.0f; // reduce precision
        write_imageui (output_lap, pos[i_line], convert_uint4(as_ushort4(convert_uchar8(tmp_data))));
    }

    result_cur[0] = result_cur[0] + result_cur[1];
    float4 final_g = (result_cur[0].even + result_cur[0].odd) * 0.25f;
    write_imageui (output_gauss, (int2)(g_x, g_y), convert_uint4(final_g));
}

/*
 * input_gauss: RGBA-CL_UNSIGNED_INT18
 * input_lap: RGBA-CL_UNSIGNED_INT16
 * output:     RGBA-CL_UNSIGNED_INT16
 * each work-item calc 2 lines
 */
__kernel void
kernel_gauss_lap_reconstruct (
    __read_only image2d_t input_gauss, __read_only image2d_t input_lap,
    __write_only image2d_t output, int out_width, int out_height)
{
    int g_x = get_global_id (0);
    int g_y = get_global_id (1);
    const sampler_t lap_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    const sampler_t gauss_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    float8 lap = convert_float8(as_uchar8(convert_ushort4(read_imageui(input_lap, lap_sampler, (int2)(g_x, g_y)))));
    lap = (lap - 128.0f) * 2.0f;

    float8 data_g;
    float pos_g_step = (1.0f / fixed_pixels) / convert_float(out_width);
    float2 pos_g_start = (float2)((g_x / convert_float(out_width)), (g_y / convert_float(out_height))) + Y_GAUSS_SCALE_OFFSSET;

    data_g.s0 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x, g_y)).x * 256.0f;
    data_g.s1 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x + pos_g_step * 1.0f, g_y)).x * 256.0f;
    data_g.s2 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x + pos_g_step * 2.0f, g_y)).x * 256.0f;
    data_g.s3 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x + pos_g_step * 3.0f, g_y)).x * 256.0f;
    data_g.s4 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x + pos_g_step * 4.0f, g_y)).x * 256.0f;
    data_g.s5 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x + pos_g_step * 5.0f, g_y)).x * 256.0f;
    data_g.s6 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x + pos_g_step * 6.0f, g_y)).x * 256.0f;
    data_g.s7 = read_imagef(input_gauss, gauss_sampler, (int2)(g_x + pos_g_step * 7.0f, g_y)).x * 256.0f;

    data_g += lap;
    write_imageui (output, (int2)(g_x, g_y), convert_uint4(as_ushort4(convert_uchar8(data_g))));
}

