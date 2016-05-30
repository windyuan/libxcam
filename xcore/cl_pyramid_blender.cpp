/*
 * cl_pyramid_blender.cpp - CL multi-band blender
 *
 *  Copyright (c) 2016 Intel Corporation
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
#include "cl_pyramid_blender.h"
#include <algorithm>
#include "cl_device.h"
#include "cl_image_bo_buffer.h"

namespace XCam {

enum {
    KernelLinearBlender = 0,
    KernelGauss,
};

const XCamKernelInfo kernels_info [] = {
    {
        "kernel_linear_blender",
#include "kernel_linear_blender.clx"
        , 0,
    },
    {
        "kernel_gauss",
#include "kernel_gauss.clx"
        , 0,
    }
};

PyramidLayer::PyramidLayer ()
    : blend_width (0)
    , blend_height (0)
{
    for (int i = 0; i < XCAM_CL_BLENDER_IMAGE_NUM; ++i) {
        width[i] = 0;
        height[i] = 0;
    }
}

CLPyramidBlender::CLPyramidBlender (const char *name, int layers)
    : CLImageHandler (name)
    , _layers (0)
    , _output_width (0)
    , _output_height (0)
{
    if (layers <= 1)
        _layers = 1;
    else if (layers > XCAM_CL_PYRAMID_MAX_LEVEL)
        _layers = XCAM_CL_PYRAMID_MAX_LEVEL;
    else
        _layers = (uint32_t)layers;
}

CLPyramidBlender::~CLPyramidBlender ()
{
}

void
CLPyramidBlender::set_blend_kernel (SmartPtr<CLLinearBlenderKernel> kernel, int index)
{
    _blend_kernels[index] = kernel;
    SmartPtr<CLImageKernel> image_kernel = kernel;
    XCAM_ASSERT (image_kernel.ptr ());
    add_kernel (image_kernel);
}

void
CLPyramidBlender::set_merge_window (const Rect &window, int layer) {
    XCAM_ASSERT (layer >= 0 && layer < XCAM_CL_PYRAMID_MAX_LEVEL);
    _pyramid_layers[layer].merge_window = window;
    XCAM_LOG_INFO(
        "Pyramid blender merge window:(x:%d, width:%d), blend_width:%d",
        window.pos_x, window.width,
        _pyramid_layers[layer].blend_width);
}

bool
CLPyramidBlender::calculate_merge_window (
    uint32_t width0, uint32_t width1, uint32_t blend_width,
    Rect &out_window)
{
    out_window.pos_x = blend_width - width1;
    out_window.width = (width0 + width1 - blend_width) / 2;

    out_window.pos_x = XCAM_ALIGN_DOWN (out_window.pos_x, 8);
    out_window.width = XCAM_ALIGN_DOWN (out_window.width, 8);
    XCAM_ASSERT (out_window.width <= blend_width);
    XCAM_ASSERT (out_window.pos_x <= blend_width);

    return true;
}

XCamReturn
CLPyramidBlender::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t output_width = _output_width;
    uint32_t output_height = input.height;
    output.init (
        input.format, output_width, output_height,
        XCAM_ALIGN_UP(output_width, 16), XCAM_ALIGN_UP(output_height, 16));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidBlender::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = CLImageHandler::prepare_output_buf (input, output);
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "CLPyramidBlender does NOT find second buffer in attachment");

    SmartPtr<DrmBoBuffer> input1 = input->find_typed_attach<DrmBoBuffer> ();
    XCAM_FAIL_RETURN(
        WARNING,
        input1.ptr (),
        XCAM_RETURN_ERROR_PARAM,
        "CLPyramidBlender does NOT find second buffer in attachment");

    ret = allocate_cl_buffers (input, input1, output);
    return ret;
}

XCamReturn
CLPyramidBlender::allocate_cl_buffers (
    SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output)
{
    uint32_t index = 0;
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();


    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (in0_info.width + in1_info.width >= _output_width);
    XCAM_ASSERT (_output_height == in0_info.height);

    _pyramid_layers[0].width[0] = in0_info.width;
    _pyramid_layers[0].height[0] = in0_info.height;
    _pyramid_layers[0].width[1] = in1_info.width;
    _pyramid_layers[0].height[1] = in1_info.height;
    _pyramid_layers[0].blend_width = _output_width;
    _pyramid_layers[0].blend_height = _output_height;

    CLImageDesc cl_desc;

    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;
    cl_desc.width = in0_info.width / 8;
    cl_desc.height = in0_info.height;
    cl_desc.row_pitch = in0_info.strides[0];
    _pyramid_layers[0].gauss_image[0] = new CLVaImage (context, input0, cl_desc, in0_info.offsets[0]);

    cl_desc.width = in1_info.width / 8;
    cl_desc.height = in1_info.height;
    cl_desc.row_pitch = in1_info.strides[0];
    _pyramid_layers[0].gauss_image[1] = new CLVaImage (context, input1, cl_desc, in1_info.offsets[0]);

    cl_desc.width = out_info.width / 8;
    cl_desc.height = out_info.height;
    cl_desc.row_pitch = out_info.strides[0];
    _pyramid_layers[0].blend_image = new CLVaImage (context, output, cl_desc, out_info.offsets[0]);

    for (index = 1; index < _layers; ++index) {
        CLImageDesc cl_desc_set;

        _pyramid_layers[index].width[0] = (_pyramid_layers[index - 1].width[0] + 1) / 2;
        _pyramid_layers[index].height[0] = (_pyramid_layers[index - 1].height[0] + 1) / 2;
        _pyramid_layers[index].width[1] = (_pyramid_layers[index - 1].width[1] + 1) / 2;
        _pyramid_layers[index].height[1] = (_pyramid_layers[index - 1].height[1] + 1) / 2;
        _pyramid_layers[index].blend_width = (_pyramid_layers[index - 1].blend_width + 1) / 2;
        _pyramid_layers[index].blend_height = (_pyramid_layers[index - 1].blend_height + 1) / 2;

        cl_desc_set.format.image_channel_data_type = CL_UNSIGNED_INT16;
        cl_desc_set.format.image_channel_order = CL_RGBA;
        cl_desc_set.row_pitch = 0;
        cl_desc_set.width = _pyramid_layers[index].width[0] / 8;
        cl_desc_set.height = _pyramid_layers[index].height[0];
        _pyramid_layers[index].gauss_image[0] = new CLImage2D (context, cl_desc_set);
        if (index != _layers - 1)
            _pyramid_layers[index].lap_image[0] = new CLImage2D (context, cl_desc_set);

        cl_desc_set.width = _pyramid_layers[index].width[1] / 8;
        cl_desc_set.height = _pyramid_layers[index].height[1];
        _pyramid_layers[index].gauss_image[1] = new CLImage2D (context, cl_desc_set);
        if (index != _layers - 1)
            _pyramid_layers[index].lap_image[1] = new CLImage2D (context, cl_desc_set);

        cl_desc_set.width = _pyramid_layers[index].blend_width / 8;
        cl_desc_set.height = _pyramid_layers[index].blend_height;
        _pyramid_layers[index].blend_image = new CLImage2D (context, cl_desc_set);

        XCAM_ASSERT (
            _pyramid_layers[index].gauss_image[0]->is_valid () &&
            _pyramid_layers[index].gauss_image[1]->is_valid () &&
            (index == (_layers - 1) || _pyramid_layers[index].lap_image[0]->is_valid ()) &&
            (index == (_layers - 1) || _pyramid_layers[index].lap_image[1]->is_valid ()) &&
            _pyramid_layers[index].blend_image->is_valid ());
    }

    // last layer lapalacian settings
    _pyramid_layers[_layers - 1].lap_image[0] = _pyramid_layers[_layers - 1].gauss_image[0];
    _pyramid_layers[_layers - 1].lap_image[1] = _pyramid_layers[_layers - 1].gauss_image[1];

    //set merge windows
    for (index = 0; index < _layers; ++index) {
        if (has_merge_window(index))
            continue;

        Rect merge_window;
        calculate_merge_window (
            _pyramid_layers[index].width[0], _pyramid_layers[index].width[1],
            _pyramid_layers[index].blend_width, merge_window);
        set_merge_window (merge_window, index);
    }
    return XCAM_RETURN_NO_ERROR;
}

CLLinearBlenderKernel::CLLinearBlenderKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t index)
    : CLImageKernel (context, kernels_info[KernelLinearBlender].kernel_name)
    , _blender (blender)
    , _index (index)
    , _blend_width (0)
{
    for (int i = 0; i < XCAM_CL_BLENDER_IMAGE_NUM; ++i)
        _input_width[i] = 0;
}

XCamReturn
CLLinearBlenderKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (output);
    SmartPtr<CLContext> context = get_context ();

    PyramidLayer &cur_layer = _blender->_pyramid_layers[_index];

    SmartPtr<CLImage> image_in0 = get_input_0(_index);
    SmartPtr<CLImage> image_in1 = get_input_1(_index);
    SmartPtr<CLImage> image_out = get_ouput (_index);
    const CLImageDesc &cl_desc_in0 = image_in0->get_image_desc ();
    const CLImageDesc &cl_desc_in1 = image_in1->get_image_desc ();
    const CLImageDesc &cl_desc_out = image_out->get_image_desc ();

    XCAM_ASSERT (cur_layer.blend_width <= cur_layer.width[0] + cur_layer.width[1]);
    _merge_window = get_merge_window (_index);
    _merge_window.pos_y = 0;
    _merge_window.height = cur_layer.blend_height;

    _blend_width = cl_desc_out.width;
    _input_width[0] = cl_desc_in0.width;
    _input_width[1] = cl_desc_in1.width;

    arg_count = 0;
    args[arg_count].arg_adress = &image_in0->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_in1->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_out->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_merge_window;
    args[arg_count].arg_size = sizeof (_merge_window);
    ++arg_count;

    args[arg_count].arg_adress = &_blend_width;
    args[arg_count].arg_size = sizeof (_blend_width);
    ++arg_count;

    args[arg_count].arg_adress = &_input_width;
    args[arg_count].arg_size = sizeof (_input_width);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 2;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLLinearBlenderKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    return CLImageKernel::post_execute (output);
}

static XCamReturn
load_kernel(SmartPtr<CLImageKernel> kernel, const XCamKernelInfo& info, const char* options = NULL)
{
    const char *body = info.kernel_body;
    return kernel->load_from_source (body, strlen (body), NULL, NULL, options);
}

SmartPtr<CLImageHandler>
create_pyramid_blender (SmartPtr<CLContext> &context, int layer)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLPyramidBlender> blender;

    XCAM_FAIL_RETURN (
        ERROR,
        layer > 0 && layer <= XCAM_CL_PYRAMID_MAX_LEVEL,
        NULL,
        "create_pyramid_blender failed with wrong layer:%d, please set it between %d and %d",
        layer, 1, XCAM_CL_PYRAMID_MAX_LEVEL);

    blender = new CLPyramidBlender ("cl_pyramid_blender", layer);
    XCAM_ASSERT (blender.ptr ());

    for (int i = 0; i < layer; ++i) {
        SmartPtr<CLLinearBlenderKernel> linear_blend_kernel;
        linear_blend_kernel = new CLLinearBlenderKernel(context, blender, (uint32_t)i);
        XCAM_ASSERT (linear_blend_kernel.ptr ());
        ret = load_kernel (linear_blend_kernel, kernels_info[KernelLinearBlender]);
        XCAM_FAIL_RETURN (
            ERROR,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "build linear blender kernel failed");
        blender->set_blend_kernel (linear_blend_kernel, i);
    }

    return blender;
}

}
