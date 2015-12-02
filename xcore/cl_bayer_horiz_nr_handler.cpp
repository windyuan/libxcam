/*
 * cl_bayer_horiz_nr_handler.cpp - CL bayer horizatal NR handler
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
#include "cl_bayer_horiz_nr_handler.h"

#define XCAM_ROTATION_GROUP_X_SIZE 64
#define XCAM_ROTATION_GROUP_Y_SIZE 64
#define ROTATION_IMAGE_PIXEL_SIZE 4

namespace XCam {

CLBayerHorizNRImageKernel::CLBayerHorizNRImageKernel (SmartPtr<CLContext> &context, uint32_t index)
    : CLImageKernel (context, "kernel_bayer_horiz_nr")
    , _component_index (index)
{
}

XCamReturn
CLBayerHorizNRImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();
    CLImageDesc in_cl_image_info, out_cl_image_info;

    in_cl_image_info.format.image_channel_order = CL_RGBA;
    in_cl_image_info.format.image_channel_data_type = CL_UNSIGNED_INT32;
    in_cl_image_info.width = in_video_info.aligned_width / 8;
    in_cl_image_info.height = in_video_info.height;
    in_cl_image_info.row_pitch = in_video_info.strides[0];

    out_cl_image_info.format.image_channel_order = CL_RGBA;
    out_cl_image_info.format.image_channel_data_type = CL_UNSIGNED_INT32;
    out_cl_image_info.width = out_video_info.width  / 8;
    out_cl_image_info.height = out_video_info.aligned_height;
    out_cl_image_info.row_pitch = out_video_info.strides[0];

    _image_in = new CLVaImage (context, input, in_cl_image_info, in_video_info.offsets[_component_index]);
    _image_out = new CLVaImage (context, output, out_cl_image_info, out_video_info.offsets[_component_index]);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    arg_count = 2;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_cl_image_info.width;
    work_size.global[1] = out_cl_image_info.height;
    work_size.local[0] = out_cl_image_info.width;
    work_size.local[1] = 1;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLBayerHorizNRImageKernel::post_execute ()
{
    return CLImageKernel::post_execute ();
}

CLBayerRotationImageKernel::CLBayerRotationImageKernel (SmartPtr<CLContext> &context, uint32_t index)
    : CLImageKernel (context, "kernel_rotation")
    , _component_index (index)
{
}

XCamReturn
CLBayerRotationImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();
    CLImageDesc in_cl_image_info, out_cl_image_info;

    in_cl_image_info.format.image_channel_order = CL_RGBA;
    in_cl_image_info.format.image_channel_data_type = CL_UNSIGNED_INT16;
    in_cl_image_info.width = in_video_info.aligned_width / ROTATION_IMAGE_PIXEL_SIZE;
    in_cl_image_info.height = in_video_info.aligned_height;
    in_cl_image_info.row_pitch = in_video_info.strides[0];

    out_cl_image_info.format.image_channel_order = CL_RGBA;
    out_cl_image_info.format.image_channel_data_type = CL_UNSIGNED_INT16;
    out_cl_image_info.width = out_video_info.width  / ROTATION_IMAGE_PIXEL_SIZE;
    out_cl_image_info.height = out_video_info.aligned_height;
    out_cl_image_info.row_pitch = out_video_info.strides[0];

    _image_in = new CLVaImage (context, input, in_cl_image_info, in_video_info.offsets[_component_index]);
    _image_out = new CLVaImage (context, output, out_cl_image_info, out_video_info.offsets[_component_index]);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    arg_count = 2;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = XCAM_ROTATION_GROUP_X_SIZE / ROTATION_IMAGE_PIXEL_SIZE;
    work_size.local[1] = XCAM_ROTATION_GROUP_Y_SIZE / ROTATION_IMAGE_PIXEL_SIZE;

    work_size.global[0] = XCAM_ALIGN_UP (in_video_info.width , XCAM_ROTATION_GROUP_X_SIZE) / XCAM_ROTATION_GROUP_X_SIZE * work_size.local[0];
    work_size.global[1] = XCAM_ALIGN_UP (in_video_info.height, XCAM_ROTATION_GROUP_Y_SIZE) / XCAM_ROTATION_GROUP_Y_SIZE * work_size.local[1];

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLBayerRotationImageKernel::post_execute ()
{
    return CLImageKernel::post_execute ();
}


SmartPtr<CLImageHandler>
create_cl_bayer_horiz_nr_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> bayer_horiz_nr_handler;
    SmartPtr<CLImageKernel> bayer_horiz_nr_kernel_0, bayer_horiz_nr_kernel_1, bayer_horiz_nr_kernel_2, bayer_horiz_nr_kernel_3;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    bayer_horiz_nr_kernel_0 = new CLBayerHorizNRImageKernel (context, 0);
    bayer_horiz_nr_kernel_1 = new CLBayerHorizNRImageKernel (context, 1);
    bayer_horiz_nr_kernel_2 = new CLBayerHorizNRImageKernel (context, 2);
    bayer_horiz_nr_kernel_3 = new CLBayerHorizNRImageKernel (context, 3);
    {
        uint32_t tmp_ret = 0;
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_bayer_horiz_nr)
#include "kernel_bayer_horiz_nr.clx"
        XCAM_CL_KERNEL_FUNC_END;
        tmp_ret = (uint32_t)bayer_horiz_nr_kernel_0->load_from_source (kernel_bayer_horiz_nr_body, strlen (kernel_bayer_horiz_nr_body));
        tmp_ret |= (uint32_t)bayer_horiz_nr_kernel_1->load_from_source (kernel_bayer_horiz_nr_body, strlen (kernel_bayer_horiz_nr_body));
        tmp_ret |= (uint32_t)bayer_horiz_nr_kernel_2->load_from_source (kernel_bayer_horiz_nr_body, strlen (kernel_bayer_horiz_nr_body));;
        tmp_ret |= (uint32_t)bayer_horiz_nr_kernel_3->load_from_source (kernel_bayer_horiz_nr_body, strlen (kernel_bayer_horiz_nr_body));;
        ret = (XCamReturn)tmp_ret;
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", bayer_horiz_nr_kernel_0->get_kernel_name());
    }
    XCAM_ASSERT (bayer_horiz_nr_kernel_0->is_valid ());
    XCAM_ASSERT (bayer_horiz_nr_kernel_1->is_valid ());
    XCAM_ASSERT (bayer_horiz_nr_kernel_2->is_valid ());
    XCAM_ASSERT (bayer_horiz_nr_kernel_3->is_valid ());
    bayer_horiz_nr_handler = new CLImageHandler ("cl_handler_bayer_horiz_nr");
    bayer_horiz_nr_handler->add_kernel  (bayer_horiz_nr_kernel_0);
    bayer_horiz_nr_handler->add_kernel  (bayer_horiz_nr_kernel_1);
    bayer_horiz_nr_handler->add_kernel  (bayer_horiz_nr_kernel_2);
    bayer_horiz_nr_handler->add_kernel  (bayer_horiz_nr_kernel_3);

    return bayer_horiz_nr_handler;
}

CLRotationHandler::CLRotationHandler (const char *name)
    : CLImageHandler (name)
    //, _output_format (XCAM_PIX_FMT_RGB48_planar)
{
}

XCamReturn
CLRotationHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t format = input.format;
    uint32_t width = input.height;
    uint32_t height = input.width;

    bool format_inited = output.init (format, width, height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) ouput format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (format));

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_bayer_rotation_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> bayer_horiz_nr_handler;
    SmartPtr<CLImageKernel> bayer_rotation_kernel_0, bayer_rotation_kernel_1, bayer_rotation_kernel_2, bayer_rotation_kernel_3;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    bayer_rotation_kernel_0 = new CLBayerRotationImageKernel (context, 0);
    bayer_rotation_kernel_1 = new CLBayerRotationImageKernel (context, 1);
    bayer_rotation_kernel_2 = new CLBayerRotationImageKernel (context, 2);
    bayer_rotation_kernel_3 = new CLBayerRotationImageKernel (context, 3);
    {
        uint32_t tmp_ret = 0;
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_rotation)
#include "kernel_rotation.clx"
        XCAM_CL_KERNEL_FUNC_END;
        tmp_ret = (uint32_t)bayer_rotation_kernel_0->load_from_source (kernel_rotation_body, strlen (kernel_rotation_body));
        tmp_ret |= (uint32_t)bayer_rotation_kernel_1->load_from_source (kernel_rotation_body, strlen (kernel_rotation_body));
        tmp_ret |= (uint32_t)bayer_rotation_kernel_2->load_from_source (kernel_rotation_body, strlen (kernel_rotation_body));;
        tmp_ret |= (uint32_t)bayer_rotation_kernel_3->load_from_source (kernel_rotation_body, strlen (kernel_rotation_body));;
        ret = (XCamReturn)tmp_ret;
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", bayer_rotation_kernel_0->get_kernel_name());
    }
    XCAM_ASSERT (bayer_rotation_kernel_0->is_valid ());
    XCAM_ASSERT (bayer_rotation_kernel_1->is_valid ());
    XCAM_ASSERT (bayer_rotation_kernel_2->is_valid ());
    XCAM_ASSERT (bayer_rotation_kernel_3->is_valid ());
    bayer_horiz_nr_handler = new CLRotationHandler ("cl_handler_bayer_rotation");
    bayer_horiz_nr_handler->add_kernel  (bayer_rotation_kernel_0);
    bayer_horiz_nr_handler->add_kernel  (bayer_rotation_kernel_1);
    bayer_horiz_nr_handler->add_kernel  (bayer_rotation_kernel_2);
    bayer_horiz_nr_handler->add_kernel  (bayer_rotation_kernel_3);

    return bayer_horiz_nr_handler;
}


};
