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
    KernelPyramidTransform,
    KernelPyramidReconstruct,
};

const XCamKernelInfo kernels_info [] = {
    {
        "kernel_linear_blender",
#include "kernel_linear_blender.clx"
        , 0,
    },
    {
        "kernel_gauss_lap_transform",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_gauss_lap_reconstruct",
#include "kernel_gauss_lap_pyramid.clx"
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

CLPyramidBlender::CLPyramidBlender (const char *name, int layers, bool need_uv)
    : CLImageHandler (name)
    , _layers (0)
    , _output_width (0)
    , _output_height (0)
    , _need_uv (need_uv)
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

SmartPtr<CLImage>
CLPyramidBlender::get_gauss_image (uint32_t layer, uint32_t buf_index, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    XCAM_ASSERT (buf_index < XCAM_CL_BLENDER_IMAGE_NUM);

    if (is_uv)
        return _pyramid_layers[layer].gauss_image_uv[buf_index];
    return _pyramid_layers[layer].gauss_image[buf_index];
}

SmartPtr<CLImage>
CLPyramidBlender::get_lap_image (uint32_t layer, uint32_t buf_index, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    XCAM_ASSERT (buf_index < XCAM_CL_BLENDER_IMAGE_NUM);

    if (is_uv)
        return _pyramid_layers[layer].lap_image_uv[buf_index];
    return _pyramid_layers[layer].lap_image[buf_index];
}

SmartPtr<CLImage>
CLPyramidBlender::get_blend_image (uint32_t layer, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);

    if (is_uv)
        return _pyramid_layers[layer].blend_image_uv[BlendImageIndex];
    return _pyramid_layers[layer].blend_image[BlendImageIndex];
}

SmartPtr<CLImage>
CLPyramidBlender::get_reconstruct_image (uint32_t layer, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);

    if (is_uv)
        return _pyramid_layers[layer].blend_image_uv[ReconstructImageIndex];
    return _pyramid_layers[layer].blend_image[ReconstructImageIndex];
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

void
PyramidLayer::bind_buf_to_image (
    SmartPtr<CLContext> context,
    SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1,
    SmartPtr<DrmBoBuffer> &output, bool last_layer, bool need_uv)
{
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    int max_plane = (need_uv ? 2 : 1);
    uint32_t divider_vert[2] = {1, 2};
    SmartPtr<CLImage> *gauss[2] = {this->gauss_image, this->gauss_image_uv};
    SmartPtr<CLImage> *lap[2] = {this->lap_image, this->lap_image_uv};
    SmartPtr<CLImage> *blend[2] = {this->blend_image, this->blend_image_uv};

    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (in0_info.width + in1_info.width >= out_info.width);
    XCAM_ASSERT (out_info.height == in0_info.height);

    this->width[0] = in0_info.width;
    this->height[0] = in0_info.height;
    this->width[1] = in1_info.width;
    this->height[1] = in1_info.height;
    this->blend_width = out_info.width;
    this->blend_height = out_info.height;

    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;

    for (int i_plane = 0; i_plane < max_plane; ++i_plane) {
        cl_desc.width = in0_info.width / 8;
        cl_desc.height = in0_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = in0_info.strides[i_plane];
        gauss[i_plane][0] = new CLVaImage (context, input0, cl_desc, in0_info.offsets[i_plane]);
        if (!last_layer)
            lap[i_plane][0] = new CLImage2D (context, cl_desc);

        cl_desc.width = in1_info.width / 8;
        cl_desc.height = in1_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = in1_info.strides[i_plane];
        gauss[i_plane][1] = new CLVaImage (context, input1, cl_desc, in1_info.offsets[i_plane]);
        if (!last_layer)
            lap[i_plane][1] = new CLImage2D (context, cl_desc);

        cl_desc.width = out_info.width / 8;
        cl_desc.height = out_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = out_info.strides[i_plane];

        blend[i_plane][ReconstructImageIndex] = new CLVaImage (context, output, cl_desc, out_info.offsets[i_plane]);
        if (!last_layer)
            blend[i_plane][BlendImageIndex] = new CLImage2D (context, cl_desc);
    }

}

void PyramidLayer::build_cl_images (SmartPtr<CLContext> context, bool last_layer, bool need_uv)
{
    CLImageDesc cl_desc_set;
    cl_desc_set.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc_set.format.image_channel_order = CL_RGBA;

    for (int i_image = 0; i_image < XCAM_CL_BLENDER_IMAGE_NUM; ++i_image) {
        SmartPtr<CLBuffer> cl_buf;
        uint32_t size = 0, row_pitch = 0;
        cl_desc_set.row_pitch = 0;
        cl_desc_set.width = this->width[i_image] / 8;
        cl_desc_set.height = this->height[i_image];

        //gauss y image created by cl buffer
        row_pitch = CLImage::calculate_pixel_bytes (cl_desc_set.format) * cl_desc_set.width;
        size = row_pitch * cl_desc_set.height;
        cl_buf = new CLBuffer (context, size);
        XCAM_ASSERT (cl_buf.ptr () && cl_buf->is_valid ());
        cl_desc_set.row_pitch = row_pitch;
        this->gauss_image[i_image] = new CLImage2D (context, cl_desc_set, 0, cl_buf);
        XCAM_ASSERT (this->gauss_image[i_image].ptr ());
        cl_desc_set.row_pitch = 0; // reset to zero

        if (!last_layer) {
            this->lap_image[i_image] = new CLImage2D (context, cl_desc_set);
            XCAM_ASSERT (this->lap_image[i_image].ptr ());
        }

        if (!need_uv)
            continue;

        cl_desc_set.height /= 2;
        //gauss uv image created by cl buffer
        size = row_pitch * cl_desc_set.height;
        cl_buf = new CLBuffer (context, size);
        XCAM_ASSERT (cl_buf.ptr () && cl_buf->is_valid ());
        cl_desc_set.row_pitch = row_pitch;
        this->gauss_image_uv[i_image] = new CLImage2D (context, cl_desc_set, 0, cl_buf);
        XCAM_ASSERT (this->gauss_image_uv[i_image].ptr ());
        if (!last_layer) {
            this->lap_image_uv[i_image] = new CLImage2D (context, cl_desc_set);
            XCAM_ASSERT (this->lap_image_uv[i_image].ptr ());
        }
    }

    cl_desc_set.width = this->blend_width / 8;
    cl_desc_set.height = this->blend_height;
    this->blend_image[ReconstructImageIndex] = new CLImage2D (context, cl_desc_set);
    XCAM_ASSERT (this->blend_image[ReconstructImageIndex].ptr ());
    if (!last_layer) {
        this->blend_image[BlendImageIndex] = new CLImage2D (context, cl_desc_set);
        XCAM_ASSERT (this->blend_image[BlendImageIndex].ptr ());
    }

    if (need_uv) {
        cl_desc_set.height /= 2;
        this->blend_image_uv[BlendImageIndex] = new CLImage2D (context, cl_desc_set);
        if (!last_layer) {
            this->blend_image_uv[ReconstructImageIndex] = new CLImage2D (context, cl_desc_set);
            XCAM_ASSERT (this->blend_image_uv[ReconstructImageIndex].ptr ());
        }
    }
}

void
CLPyramidBlender::last_layer_buffer_redirect ()
{
    PyramidLayer &layer = _pyramid_layers[_layers - 1];

    layer.blend_image[BlendImageIndex] = layer.blend_image[ReconstructImageIndex];
    if (_need_uv)
        layer.blend_image_uv[BlendImageIndex] = layer.blend_image_uv[ReconstructImageIndex];

    for (uint32_t i_image = 0; i_image < XCAM_CL_BLENDER_IMAGE_NUM; ++i_image) {
        layer.lap_image[i_image] = layer.gauss_image[i_image];
        if (_need_uv)
            layer.lap_image_uv[i_image] = layer.gauss_image_uv[i_image];
    }
}

XCamReturn
CLPyramidBlender::allocate_cl_buffers (
    SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output)
{
    uint32_t index = 0;
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();

    _pyramid_layers[0].bind_buf_to_image (context, input0, input1, output, (0 == _layers - 1), _need_uv);

    for (index = 1; index < _layers; ++index) {
        _pyramid_layers[index].width[0] = (_pyramid_layers[index - 1].width[0] + 1) / 2;
        _pyramid_layers[index].height[0] = (_pyramid_layers[index - 1].height[0] + 1) / 2;
        _pyramid_layers[index].width[1] = (_pyramid_layers[index - 1].width[1] + 1) / 2;
        _pyramid_layers[index].height[1] = (_pyramid_layers[index - 1].height[1] + 1) / 2;
        _pyramid_layers[index].blend_width = (_pyramid_layers[index - 1].blend_width + 1) / 2;
        _pyramid_layers[index].blend_height = (_pyramid_layers[index - 1].blend_height + 1) / 2;

        _pyramid_layers[index].build_cl_images (context, (index == _layers - 1), _need_uv);
    }

    //last layer buffer redirect
    last_layer_buffer_redirect ();

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
    uint32_t layer, bool is_uv)
    : CLImageKernel (context, kernels_info[KernelLinearBlender].kernel_name)
    , _blender (blender)
    , _is_uv (is_uv)
    , _layer (layer)
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
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    SmartPtr<CLContext> context = get_context ();

    PyramidLayer &cur_layer = _blender->_pyramid_layers[_layer];

    SmartPtr<CLImage> image_in0 = get_input_0();
    SmartPtr<CLImage> image_in1 = get_input_1();
    SmartPtr<CLImage> image_out = get_ouput ();
    const CLImageDesc &cl_desc_in0 = image_in0->get_image_desc ();
    const CLImageDesc &cl_desc_in1 = image_in1->get_image_desc ();
    const CLImageDesc &cl_desc_out = image_out->get_image_desc ();

    XCAM_ASSERT (cur_layer.blend_width <= cur_layer.width[0] + cur_layer.width[1]);
    _merge_window = get_merge_window ();
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

CLPyramidTransformKernel::CLPyramidTransformKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    uint32_t buf_index,
    bool is_uv)
    : CLImageKernel (context, kernels_info[KernelPyramidTransform].kernel_name)
    , _blender (blender)
    , _layer (layer)
    , _buf_index (buf_index)
    , _is_uv (is_uv)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
    XCAM_ASSERT (buf_index <= XCAM_CL_BLENDER_IMAGE_NUM);
}

bool change_image_format (
    SmartPtr<CLContext> context, SmartPtr<CLImage> input, SmartPtr<CLImage> &output, const CLImageDesc &new_desc)
{
    SmartPtr<CLImage2D> previous = input.dynamic_cast_ptr<CLImage2D> ();
    if (!previous.ptr () || !previous->get_bind_buf ().ptr ())
        return false;

    SmartPtr<CLBuffer> bind_buf = previous->get_bind_buf ();
    output = new CLImage2D (context, new_desc, 0, bind_buf);
    if (!output.ptr ())
        return false;
    return true;
}

XCamReturn
CLPyramidTransformKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in_gauss = get_input_gauss();
    SmartPtr<CLImage> image_out_gauss = get_output_gauss();
    SmartPtr<CLImage> image_out_lap = get_output_lap ();
    const CLImageDesc &cl_desc_out_gauss_pre = image_out_gauss->get_image_desc ();

    CLImageDesc cl_desc_out_gauss;
    cl_desc_out_gauss.format.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_desc_out_gauss.format.image_channel_order = CL_RGBA;
    cl_desc_out_gauss.width = cl_desc_out_gauss_pre.width * 2;
    cl_desc_out_gauss.height = cl_desc_out_gauss_pre.height;
    cl_desc_out_gauss.row_pitch = cl_desc_out_gauss_pre.row_pitch;
    _output_gauss.release ();
    change_image_format (context, image_out_gauss, _output_gauss, cl_desc_out_gauss);
    XCAM_FAIL_RETURN (
        ERROR,
        _output_gauss.ptr () && _output_gauss->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change ouput gauss image format failed");

    arg_count = 0;
    args[arg_count].arg_adress = &image_in_gauss->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_output_gauss->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_out_lap->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 2;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out_gauss.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out_gauss.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidTransformKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _output_gauss.release ();
    return CLImageKernel::post_execute (output);
}

CLPyramidReconstructKernel::CLPyramidReconstructKernel (
    SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, bool is_uv)
    : CLImageKernel (context, kernels_info[KernelPyramidReconstruct].kernel_name)
    , _blender (blender)
    , _layer (layer)
    , _is_uv (is_uv)
    , _out_gauss_width (0)
    , _out_gauss_height (0)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
}

XCamReturn
CLPyramidReconstructKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in_gauss = get_input_gauss();
    SmartPtr<CLImage> image_in_lap = get_input_lap ();
    SmartPtr<CLImage> image_out_gauss = get_output_gauss();
    const CLImageDesc &cl_desc_in_gauss_pre = image_in_gauss->get_image_desc ();
    const CLImageDesc &cl_desc_out_gauss = image_out_gauss->get_image_desc ();

    CLImageDesc cl_desc_in_gauss;
    cl_desc_in_gauss.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc_in_gauss.format.image_channel_order = CL_R;
    cl_desc_in_gauss.width = cl_desc_in_gauss_pre.width * 8;
    cl_desc_in_gauss.height = cl_desc_in_gauss_pre.height;
    cl_desc_in_gauss.row_pitch = cl_desc_in_gauss_pre.row_pitch;
    _input_gauss.release ();
    change_image_format (context, image_in_gauss, _input_gauss, cl_desc_in_gauss);
    XCAM_FAIL_RETURN (
        ERROR,
        _input_gauss.ptr () && _input_gauss->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change ouput gauss image format failed");

    _out_gauss_width = cl_desc_out_gauss.width;
    _out_gauss_height = cl_desc_out_gauss.height;

    arg_count = 0;
    args[arg_count].arg_adress = &_input_gauss->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_in_lap->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_out_gauss->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_out_gauss_width;
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_out_gauss_height;
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 2;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out_gauss.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out_gauss.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidReconstructKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _input_gauss.release ();
    return CLImageKernel::post_execute (output);
}

static XCamReturn
load_kernel(SmartPtr<CLKernel> kernel, const XCamKernelInfo& info, const char* options = NULL)
{
    const char *body = info.kernel_body;
    return kernel->load_from_source (body, strlen (body), NULL, NULL, options);
}

static SmartPtr<CLImageKernel>
create_linear_blend_kernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    bool is_uv)
{
    static SmartPtr<CLKernel> linear_blend_kernel[2]; // 0-Y, 1-UV
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    uint32_t yuv_index = (is_uv ? 0 : 1);

    SmartPtr<CLImageKernel> kernel;

    if (!linear_blend_kernel[yuv_index].ptr ()) {
        SmartPtr<CLKernel> tmp_kernel;
        char blend_option[1024];
        snprintf (blend_option, sizeof(blend_option), "-DBLEND_UV=%d", yuv_index);
        tmp_kernel = new CLKernel(context, kernels_info[KernelLinearBlender].kernel_name);
        XCAM_ASSERT (tmp_kernel.ptr ());
        ret = load_kernel (tmp_kernel, kernels_info[KernelLinearBlender], blend_option);
        XCAM_FAIL_RETURN (
            ERROR,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "build linear blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
        linear_blend_kernel[yuv_index] = tmp_kernel;
    }

    kernel = new CLLinearBlenderKernel(context, blender, layer, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    ret = kernel->load_from_kernel(linear_blend_kernel[yuv_index]);
    XCAM_FAIL_RETURN (
        ERROR,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "load linear blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_transform_kernel (
    SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, uint32_t buf_index, bool is_uv)
{
    static SmartPtr<CLKernel> pyramid_transform_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SmartPtr<CLImageKernel> kernel;

    if (!pyramid_transform_kernel.ptr ()) {
        SmartPtr<CLKernel> tmp_kernel;
        char transform_option[1024];
        snprintf (transform_option, sizeof(transform_option), "-DTRANSFORM_UV=%d", (is_uv ? 1 : 0));
        tmp_kernel = new CLKernel(context, kernels_info[KernelPyramidTransform].kernel_name);
        XCAM_ASSERT (tmp_kernel.ptr ());
        ret = load_kernel (tmp_kernel, kernels_info[KernelPyramidTransform], transform_option);
        XCAM_FAIL_RETURN (
            ERROR,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "build pyramid transform kernel(%s) failed", (is_uv ? "UV" : "Y"));
        pyramid_transform_kernel = tmp_kernel;
    }

    kernel = new CLPyramidTransformKernel (context, blender, layer, buf_index, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    ret = kernel->load_from_kernel(pyramid_transform_kernel);
    XCAM_FAIL_RETURN (
        ERROR,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid transform kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_reconstruct_kernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    bool is_uv)
{
    static SmartPtr<CLKernel> pyramid_reconstruct_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SmartPtr<CLImageKernel> kernel;

    if (!pyramid_reconstruct_kernel.ptr ()) {
        SmartPtr<CLKernel> tmp_kernel;
        char transform_option[1024];
        snprintf (transform_option, sizeof(transform_option), "-DTRANSFORM_UV=%d", (is_uv ? 1 : 0));
        tmp_kernel = new CLKernel(context, kernels_info[KernelPyramidReconstruct].kernel_name);
        XCAM_ASSERT (tmp_kernel.ptr ());
        ret = load_kernel (tmp_kernel, kernels_info[KernelPyramidReconstruct], transform_option);
        XCAM_FAIL_RETURN (
            ERROR,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "build pyramid reconstruct kernel(%s) failed", (is_uv ? "UV" : "Y"));
        pyramid_reconstruct_kernel = tmp_kernel;
    }

    kernel = new CLPyramidReconstructKernel (context, blender, layer, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    ret = kernel->load_from_kernel(pyramid_reconstruct_kernel);
    XCAM_FAIL_RETURN (
        ERROR,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid reconstruct kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}


SmartPtr<CLImageHandler>
create_pyramid_blender (SmartPtr<CLContext> &context, int layer, bool need_uv)
{
    SmartPtr<CLPyramidBlender> blender;
    SmartPtr<CLImageKernel> kernel;
    int i = 0;

    XCAM_FAIL_RETURN (
        ERROR,
        layer > 0 && layer <= XCAM_CL_PYRAMID_MAX_LEVEL,
        NULL,
        "create_pyramid_blender failed with wrong layer:%d, please set it between %d and %d",
        layer, 1, XCAM_CL_PYRAMID_MAX_LEVEL);

    blender = new CLPyramidBlender ("cl_pyramid_blender", layer, need_uv);
    XCAM_ASSERT (blender.ptr ());

    for (uint32_t buf_index = 0; buf_index < XCAM_CL_BLENDER_IMAGE_NUM; ++buf_index) {
        for (i = 0; i < layer - 1; ++i) {
            kernel = create_pyramid_transform_kernel (context, blender, (uint32_t)i, buf_index, false);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid transform kernel failed");
            blender->add_kernel (kernel);

            if (need_uv) {
                kernel = create_pyramid_transform_kernel (context, blender, (uint32_t)i, buf_index, true);
                XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid transform kernel failed");
                blender->add_kernel (kernel);
            }
        }
    }

    for (i = 0; i < layer; ++i) {
        kernel = create_linear_blend_kernel (context, blender, (uint32_t)i, false);
        XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create linear blender kernel failed");
        blender->add_kernel (kernel);

        if (need_uv) {
            kernel = create_linear_blend_kernel (context, blender, (uint32_t)i, true);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create linear blender kernel failed");
            blender->add_kernel (kernel);
        }
    }

    for (i = layer - 2; i >= 0 && i < layer; --i) {
        kernel = create_pyramid_reconstruct_kernel (context, blender, (uint32_t)i, false);
        XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid transform kernel failed");
        blender->add_kernel (kernel);

        if (need_uv) {
            kernel = create_pyramid_reconstruct_kernel (context, blender, (uint32_t)i, true);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid transform kernel failed");
            blender->add_kernel (kernel);
        }
    }

    return blender;
}

}
