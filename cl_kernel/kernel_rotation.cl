/*
 * function: kernel_rotation.cl
 *     sample code of default kernel arguments
 * input:    image2d_t as read only
 * output:   image2d_t as write only
 */

//Group pixel 64*64

// pixel = ROW_SIZE * 4
#define ROW_SIZE 16
// num of pixel rows NUM_OF_ROWS * 4
#define NUM_OF_ROWS 16

#define IMAGE_PIXEL_SIZE 4

__kernel void kernel_rotation (
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
    int input_x = g_x, input_y = g_y * IMAGE_PIXEL_SIZE;
    ushort4 tmp0, tmp1, tmp2, tmp3;
    ushort8 in_data, out_data;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

    int store_index = mad24 (l_y, l_x_size, l_x + l_y);

    __local ushort4 slm_0[ROW_SIZE * (NUM_OF_ROWS + 1)], slm_1[ROW_SIZE * (NUM_OF_ROWS + 1)],
            slm_2[ROW_SIZE * (NUM_OF_ROWS + 1)], slm_3[ROW_SIZE * (NUM_OF_ROWS + 1)];

    tmp0 = convert_ushort4 (read_imageui(input, sampler, (int2)(input_x, input_y)));
    slm_0[store_index] = tmp0;
    tmp1 = convert_ushort4 (read_imageui(input, sampler, (int2)(input_x, input_y + 1)));
    slm_1[store_index] = tmp1;
    tmp2 = convert_ushort4 (read_imageui(input, sampler, (int2)(input_x, input_y + 2)));
    slm_2[store_index] = tmp2;
    tmp3 = convert_ushort4 (read_imageui(input, sampler, (int2)(input_x, input_y + 3)));
    slm_3[store_index] = tmp3;
    barrier (CLK_LOCAL_MEM_FENCE);

    in_data = (ushort8) (tmp0.s0, tmp1.s0, tmp2.s0, tmp3.s0, tmp0.s1, tmp1.s1, tmp2.s1, tmp3.s1);

    int load_index = mad24 (l_x, l_x_size, l_x + l_y);
    tmp0 = slm_0[load_index];
    tmp1 = slm_1[load_index];
    tmp2 = slm_2[load_index];
    tmp3 = slm_3[load_index];
    int ouput_x = get_group_id(1) * ROW_SIZE + l_x;
    int ouput_y = (get_group_id(0) * ROW_SIZE + l_y) * IMAGE_PIXEL_SIZE;

#if 0
    if (l_x % 16 == 8 && l_y % 16 == 8) {
        out_data = (ushort8) (tmp0.s0, tmp1.s0, tmp2.s0, tmp3.s0, tmp0.s1, tmp1.s1, tmp2.s1, tmp3.s1);
        printf ("in(%d,%d)(%d,%d,%d,%d,%d,%d,%d,%d), out(%d, %d)(%d,%d,%d,%d,%d,%d,%d,%d)\n",
                input_x * 4, input_y,
                in_data.s0, in_data.s1, in_data.s2, in_data.s3, in_data.s4, in_data.s5, in_data.s6, in_data.s7,
                ouput_x * 4, ouput_y,
                out_data.s0, out_data.s1, out_data.s2, out_data.s3, out_data.s4, out_data.s5, out_data.s6, out_data.s7
               );
    }
#endif

    write_imageui (output, (int2)(ouput_x, ouput_y), convert_uint4 ((ushort4)(tmp0.s0, tmp1.s0, tmp2.s0, tmp3.s0)));
    write_imageui (output, (int2)(ouput_x, ouput_y + 1), convert_uint4 ((ushort4)(tmp0.s1, tmp1.s1, tmp2.s1, tmp3.s1)));
    write_imageui (output, (int2)(ouput_x, ouput_y + 2), convert_uint4 ((ushort4)(tmp0.s2, tmp1.s2, tmp2.s2, tmp3.s2)));
    write_imageui (output, (int2)(ouput_x, ouput_y + 3), convert_uint4 ((ushort4)(tmp0.s3, tmp1.s3, tmp2.s3, tmp3.s3)));
}

