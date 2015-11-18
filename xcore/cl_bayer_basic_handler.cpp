/*
 * cl_bayer_basic_handler.cpp - CL bayer basic handler
 *
 *  Copyright (c) 2015 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "xcam_utils.h"
#include "cl_bayer_basic_handler.h"

namespace XCam {

CLBayerBasicImageKernel::CLBayerBasicImageKernel (SmartPtr<CLContext> &context, SmartPtr<CLBayerBasicImageHandler>& handler)
    : CLImageKernel (context, "kernel_bayer_basic")
    , _out_aligned_height (0)
    , _handler (handler)
{
    _blc_config.level_gr = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_r = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_b = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_gb = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.color_bits = 10;

    _wb_config.r_gain = 1.0;
    _wb_config.gr_gain = 1.0;
    _wb_config.gb_gain = 1.0;
    _wb_config.b_gain = 1.0;

    for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; i++)
        _gamma_table[i] = (float)i / 256.0f;
    _gamma_table[XCAM_GAMMA_TABLE_SIZE] = 0.9999f;

    _3a_stats_context = new CL3AStatsCalculatorContext (context);
    XCAM_ASSERT (_3a_stats_context.ptr ());
}

bool
CLBayerBasicImageKernel::set_blc (const XCam3aResultBlackLevel &blc)
{
    _blc_config.level_r = (float)blc.r_level;
    _blc_config.level_gr = (float)blc.gr_level;
    _blc_config.level_gb = (float)blc.gb_level;
    _blc_config.level_b = (float)blc.b_level;
    //_blc_config.color_bits = 0;
    return true;
}

bool
CLBayerBasicImageKernel::set_wb (const XCam3aResultWhiteBalance &wb)
{
    _wb_config.r_gain = (float)wb.r_gain;
    _wb_config.gr_gain = (float)wb.gr_gain;
    _wb_config.gb_gain = (float)wb.gb_gain;
    _wb_config.b_gain = (float)wb.b_gain;
    return true;
}

bool
CLBayerBasicImageKernel::set_gamma_table (const XCam3aResultGammaTable &gamma)
{
    for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; i++)
        _gamma_table[i] = (float)gamma.table[i] / 256.0f;

    return true;
}

XCamReturn
CLBayerBasicImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();
    CLImageDesc in_image_info;
    CLImageDesc out_image_info;

    if (!_3a_stats_context->is_ready () && !_3a_stats_context->allocate_data (in_video_info)) {
        XCAM_LOG_WARNING ("CL3AStatsCalculatorContext allocate data failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    in_image_info.format.image_channel_order = CL_RGBA;
    in_image_info.format.image_channel_data_type = CL_UNSIGNED_INT32; //CL_UNORM_INT16;
    in_image_info.width = in_video_info.aligned_width / 8;
    in_image_info.height = in_video_info.height;
    in_image_info.row_pitch = in_video_info.strides[0];

    out_image_info.format.image_channel_order = CL_RGBA;
    out_image_info.format.image_channel_data_type = CL_UNSIGNED_INT16; //CL_UNORM_INT16;
    out_image_info.width = out_video_info.width  / 4;
    out_image_info.height = out_video_info.aligned_height * 4;
    out_image_info.row_pitch = out_video_info.strides[0];

    //_image_in = new CLVaImage (context, input, in_image_info);
    _buffer_in = new CLVaBuffer (context, input);
    _image_out = new CLVaImage (context, output, out_image_info);
    //_image_out = new CLVaImage (context, output);

    _out_aligned_height = out_video_info.aligned_height;
    _blc_config.color_bits = in_video_info.color_bits;

    _gamma_table_buffer = new CLBuffer(
        context, sizeof(float) * (XCAM_GAMMA_TABLE_SIZE + 1),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_gamma_table);

    _stats_cl_buffer = _3a_stats_context->get_next_buffer ();

#if 0
    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());
#endif

    //set args;
    arg_count = 0;
    //args[arg_count].arg_adress = &_image_in->get_mem_id ();
    args[arg_count].arg_adress = &_buffer_in->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_image_out->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_out_aligned_height;
    args[arg_count].arg_size = sizeof (_out_aligned_height);
    ++arg_count;

    args[arg_count].arg_adress = &_blc_config;
    args[arg_count].arg_size = sizeof (_blc_config);
    ++arg_count;

    args[arg_count].arg_adress = &_wb_config;
    args[arg_count].arg_size = sizeof (_wb_config);
    ++arg_count;

    args[arg_count].arg_adress = &_gamma_table_buffer->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_stats_cl_buffer->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_video_info.width  / 4;
    work_size.global[1] = out_video_info.aligned_height;
    work_size.local[0] = 8;
    work_size.local[1] = 4;

    //printf ("work_size:g(%d, %d), l(%d, %d)\n", work_size.global[0], work_size.global[1], work_size.local[0], work_size.local[1]);

    _output_buffer = output;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLBayerBasicImageKernel::post_execute ()
{
    SmartPtr<X3aStats> stats_3a;

    _buffer_in.release ();
    _gamma_table_buffer.release ();

    CLImageKernel::post_execute ();

    stats_3a = _3a_stats_context->copy_stats_out (_stats_cl_buffer);
    if (!stats_3a.ptr ()) {
        XCAM_LOG_DEBUG ("copy 3a stats failed, maybe handler stopped");
        return XCAM_RETURN_ERROR_CL;
    }

    stats_3a->set_timestamp (_output_buffer->get_timestamp ());
    _output_buffer->attach_buffer (stats_3a);

    _stats_cl_buffer.release ();
    _output_buffer.release ();

    XCAM_FAIL_RETURN (WARNING, stats_3a.ptr (), XCAM_RETURN_ERROR_MEM, "3a stats dequeue failed");
    //return XCAM_RETURN_NO_ERROR;
    return _handler->post_stats (stats_3a);
}

void
CLBayerBasicImageKernel::pre_stop ()
{
    _3a_stats_context->pre_stop ();
}

CLBayerBasicImageHandler::CLBayerBasicImageHandler (const char *name)
    : CLImageHandler (name)
{
}

bool
CLBayerBasicImageHandler::set_bayer_kernel (SmartPtr<CLBayerBasicImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _bayer_kernel = kernel;
    return true;
}

bool
CLBayerBasicImageHandler::set_blc_config (const XCam3aResultBlackLevel &blc)
{
    return _bayer_kernel->set_blc (blc);
}

bool
CLBayerBasicImageHandler::set_wb_config (const XCam3aResultWhiteBalance &wb)
{
    return _bayer_kernel->set_wb (wb);
}

bool
CLBayerBasicImageHandler::set_gamma_table (const XCam3aResultGammaTable &gamma)
{
    return _bayer_kernel->set_gamma_table (gamma);
}

XCamReturn
CLBayerBasicImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t format = XCAM_PIX_FMT_SGRBG16_planar;
    bool format_inited = output.init (format, input.width / 2 , input.height / 2);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) ouput format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (format));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLBayerBasicImageHandler::post_stats (const SmartPtr<X3aStats> &stats)
{
    if (_stats_callback.ptr ())
        return _stats_callback->x3a_stats_ready (stats);

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_bayer_basic_image_handler (SmartPtr<CLContext> &context, bool enable_gamma)
{
    SmartPtr<CLBayerBasicImageHandler> bayer_planar_handler;
    SmartPtr<CLBayerBasicImageKernel> basic_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    bayer_planar_handler = new CLBayerBasicImageHandler ("CLBayerBasicImageHandler");
    basic_kernel = new CLBayerBasicImageKernel (context, bayer_planar_handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN (kernel_bayer_copy)
#include "kernel_bayer_basic.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = basic_kernel->load_from_source (
                  kernel_bayer_copy_body, strlen (kernel_bayer_copy_body),
                  NULL, NULL,
                  enable_gamma ? "-DENABLE_GAMMA=1" : "-DENABLE_GAMMA=0");
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", basic_kernel->get_kernel_name());
    }
    XCAM_ASSERT (basic_kernel->is_valid ());
    bayer_planar_handler->set_bayer_kernel (basic_kernel);

    return bayer_planar_handler;
}

};
