/*
 * function: kernel_bayer_horiz_nr.cl
 *     sample code of default kernel arguments
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

//#define ENABLE_IMAGE_2D_INPUT 0

/*
 * GROUP_PIXEL_X_SIZE = 2 * GROUP_CELL_X_SIZE
 * GROUP_PIXEL_Y_SIZE = 2 * GROUP_CELL_Y_SIZE
*/

//float4; 16
#define SLM_LINE_SIZE (2048/4)
#define INDEX_OFFSET  1
#define DELTA_COEF_TABLE_SIZE 64
#define SPATIAL_COEF_TABLE_SIZE 4

__constant float g_spatial_coeff[9] = {8.0f, 7.5f, 7.2f, 7.0f, 6.8f, 6.5f, 6.0f, 5.0f, 4.0f};


#define floatX float4
#define DELTA_COEFF_SIZE 64.0f

inline floatX delta_coeff_c (floatX delta, __local float *delta_coeff)
{
    floatX ret;
    uint4 index = convert_uint4(fabs(delta) * DELTA_COEFF_SIZE);
    ret.s0 = delta_coeff[index.s0];
    ret.s1 = delta_coeff[index.s1];
    ret.s2 = delta_coeff[index.s2];
    ret.s3 = delta_coeff[index.s3];
    return ret;
}

inline void denoise (float value, floatX neighbor, floatX s_coeff, __local float *delta_coeff, float *out_value, float *out_gain)
{
    floatX coeff = s_coeff * delta_coeff_c (neighbor - value, delta_coeff);
    floatX result = coeff * neighbor;
    (*out_value) = result.s0 + result.s1 + result.s2 + result.s3,
    (*out_gain) = coeff.s0 + coeff.s1 + coeff.s2 + coeff.s3;
}

//s_coeff: position distance : 1,2,3,4,5,6,7
inline void caculate_noise_left (floatX value, floatX neighbor, float8 s_coeff, __local float *delta_coeff, floatX *sum_value, floatX *sum_gain)
{
    float tmp_sum_value, tmp_sum_gain;
    floatX fixed_coeff;
    fixed_coeff = s_coeff.s3210;
    denoise (value.s0, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s0 += tmp_sum_value;
    (*sum_gain).s0 += tmp_sum_gain;
    fixed_coeff = s_coeff.s4321;
    denoise (value.s1, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s1 += tmp_sum_value;
    (*sum_gain).s1 += tmp_sum_gain;
    fixed_coeff = s_coeff.s5432;
    denoise (value.s2, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s2 += tmp_sum_value;
    (*sum_gain).s2 += tmp_sum_gain;
    fixed_coeff = s_coeff.s6543;
    denoise (value.s3, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s3 += tmp_sum_value;
    (*sum_gain).s3 += tmp_sum_gain;
}

//s_coeff: position distance : 1,2,3,4,5,6,7
inline void caculate_noise_right (floatX value, floatX neighbor, float8 s_coeff, __local float *delta_coeff, floatX *sum_value, floatX *sum_gain)
{
    float tmp_sum_value, tmp_sum_gain;
    floatX fixed_coeff;
    fixed_coeff = s_coeff.s3456;
    denoise (value.s0, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s0 += tmp_sum_value;
    (*sum_gain).s0 += tmp_sum_gain;
    fixed_coeff = s_coeff.s2345;
    denoise (value.s1, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s1 += tmp_sum_value;
    (*sum_gain).s1 += tmp_sum_gain;
    fixed_coeff = s_coeff.s1234;
    denoise (value.s2, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s2 += tmp_sum_value;
    (*sum_gain).s2 += tmp_sum_gain;
    fixed_coeff = s_coeff.s0123;
    denoise (value.s3, neighbor, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s3 += tmp_sum_value;
    (*sum_gain).s3 += tmp_sum_gain;
}

//s_coeff: position distance : 0,1,2,3
inline void caculate_noise_self (floatX value, float4 s_coeff, __local float *delta_coeff, floatX *sum_value, floatX *sum_gain)
{
    float tmp_sum_value, tmp_sum_gain;
    floatX fixed_coeff;
    fixed_coeff = s_coeff.s0123;
    denoise (value.s0, value, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s0 += tmp_sum_value;
    (*sum_gain).s0 += tmp_sum_gain;
    fixed_coeff = s_coeff.s1012;
    denoise (value.s1, value, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s1 += tmp_sum_value;
    (*sum_gain).s1 += tmp_sum_gain;
    fixed_coeff = s_coeff.s2101;
    denoise (value.s2, value, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s2 += tmp_sum_value;
    (*sum_gain).s2 += tmp_sum_gain;
    fixed_coeff = s_coeff.s3210;
    denoise (value.s3, value, fixed_coeff, delta_coeff, &tmp_sum_value, &tmp_sum_gain);
    (*sum_value).s3 += tmp_sum_value;
    (*sum_gain).s3 += tmp_sum_gain;
}

__kernel void kernel_bayer_horiz_nr (
    __read_only image2d_t input,
    __write_only image2d_t output
)
{
    const int g_x = get_global_id (0);
    const int g_y = get_global_id (1);

    const int l_x = get_local_id (0);
    const int l_y = get_local_id (1);
    const int l_x_size = get_local_size (0);
    const int l_y_size = get_local_size (1);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    int index = mad24 (l_y, l_x_size, l_x);


    __local float4 slm_line_p1[SLM_LINE_SIZE / 2], slm_line_p2[SLM_LINE_SIZE / 2];
    __local float delta_coeff[DELTA_COEF_TABLE_SIZE];
    __local float spatial_coeff[DELTA_COEF_TABLE_SIZE];

    float8 data = convert_float8 (as_ushort8 (read_imageui(input, sampler, (int2)(g_x, g_y)))) / 65536.0f;
    slm_line_p1[index] = data.lo;
    slm_line_p2[index] = data.hi;
    barrier (CLK_LOCAL_MEM_FENCE);

    float4 sum1_value = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1_gain = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum2_value = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum2_gain = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    float4 self_coeff = (float4) (g_spatial_coeff[0], g_spatial_coeff[1], g_spatial_coeff[2], g_spatial_coeff[3]);
    float8 neiggbor_coeff = (float8) (g_spatial_coeff[1], g_spatial_coeff[2], g_spatial_coeff[3], g_spatial_coeff[4],
                                      g_spatial_coeff[5], g_spatial_coeff[6], g_spatial_coeff[7], 0.0f);

    float4 p1_cur = slm_line_p1[index];
    float4 p2_cur = slm_line_p2[index];
    caculate_noise_self (p1_cur, self_coeff, delta_coeff, &sum1_value, &sum1_gain);
    caculate_noise_right (p1_cur, p2_cur, neiggbor_coeff, delta_coeff, &sum1_value, &sum1_gain);

    caculate_noise_self (p2_cur, self_coeff, delta_coeff, &sum2_value, &sum2_gain);
    caculate_noise_left (p2_cur, p1_cur, neiggbor_coeff, delta_coeff, &sum2_value, &sum2_gain);

    float4 p2_pre = slm_line_p2[index - 1];
    caculate_noise_left (p1_cur, p2_pre, neiggbor_coeff, delta_coeff, &sum1_value, &sum1_gain);

    float4 p1_nxt = slm_line_p1[index + 1];
    caculate_noise_right (p2_cur, p1_nxt, neiggbor_coeff, delta_coeff, &sum2_value, &sum2_gain);

    float8 result = (float8)(sum1_value / sum1_gain, sum2_value / sum2_gain);

    write_imageui (output, (int2)(g_x, g_y), as_uint4 (convert_ushort8_sat (result * 65536.0f)));
}

