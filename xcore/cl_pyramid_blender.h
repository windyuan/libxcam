/*
 * cl_pyramid_blender.h - CL pyramid blender
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

#ifndef XCAM_CL_PYRAMID_BLENDER_H
#define XCAM_CL_PYRAMID_BLENDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

#define XCAM_CL_PYRAMID_MAX_LEVEL  4
#define XCAM_CL_BLENDER_IMAGE_NUM  2


namespace XCam {

class CLPyramidBlender;

struct Rect {
    uint32_t pos_x, pos_y;
    uint32_t width, height;

    Rect () : pos_x (0), pos_y (0), width (0), height (0) {}
};

enum {
    BlendImageIndex = 0,
    ReconstructImageIndex,
    BlendImageCount
};

struct PyramidLayer {
    uint32_t                 width[XCAM_CL_BLENDER_IMAGE_NUM];
    uint32_t                 height[XCAM_CL_BLENDER_IMAGE_NUM];
    uint32_t                 blend_width;
    uint32_t                 blend_height;
    Rect                     merge_window;
    SmartPtr<CLImage>        gauss_image[XCAM_CL_BLENDER_IMAGE_NUM];
    SmartPtr<CLImage>        lap_image[XCAM_CL_BLENDER_IMAGE_NUM];
    SmartPtr<CLImage>        blend_image[BlendImageCount]; // 0 blend-image, 1 reconstruct image

    SmartPtr<CLImage>        gauss_image_uv[XCAM_CL_BLENDER_IMAGE_NUM];
    SmartPtr<CLImage>        lap_image_uv[XCAM_CL_BLENDER_IMAGE_NUM];
    SmartPtr<CLImage>        blend_image_uv[BlendImageCount];

    PyramidLayer ();
    void build_cl_images (SmartPtr<CLContext> context, bool need_lap, bool need_uv);
    void bind_buf_to_image (
        SmartPtr<CLContext> context,
        SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1,
        SmartPtr<DrmBoBuffer> &output, bool last_layer, bool need_uv);
};

class CLLinearBlenderKernel;

class CLPyramidBlender
    : public CLImageHandler
{
    friend class CLLinearBlenderKernel;

public:
    explicit CLPyramidBlender (const char *name, int layers, bool need_uv);
    ~CLPyramidBlender ();
    void set_output_size (uint32_t width, uint32_t height) {
        _output_width = width;
        _output_height = height;
    }

    void set_merge_window (const Rect &window, int layer);

    bool has_merge_window (uint32_t layer) const {
        return (_pyramid_layers[layer].merge_window.width ? true : false);
    }

    //void set_blend_kernel (SmartPtr<CLLinearBlenderKernel> kernel, int index);
    SmartPtr<CLImage> get_gauss_image (uint32_t layer, uint32_t buf_index, bool is_uv);
    SmartPtr<CLImage> get_lap_image (uint32_t layer, uint32_t buf_index, bool is_uv);
    SmartPtr<CLImage> get_blend_image (uint32_t layer, bool is_uv);
    SmartPtr<CLImage> get_reconstruct_image (uint32_t layer, bool is_uv);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    XCamReturn allocate_cl_buffers (
        SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output);

private:
    bool calculate_merge_window (uint32_t width0, uint32_t width1, uint32_t blend_width, Rect &out_window);
    void last_layer_buffer_redirect ();

    XCAM_DEAD_COPY (CLPyramidBlender);

private:
    uint32_t                         _layers;
    uint32_t                         _output_width;
    uint32_t                         _output_height;
    bool                             _need_uv;
    PyramidLayer                     _pyramid_layers[XCAM_CL_PYRAMID_MAX_LEVEL];
};

class CLLinearBlenderKernel
    : public CLImageKernel
{
public:
    explicit CLLinearBlenderKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    SmartPtr<CLImage> get_input_0 () {
        return _blender->get_lap_image (_layer, 0, _is_uv);
    }
    SmartPtr<CLImage> get_input_1 () {
        return _blender->get_lap_image (_layer, 1, _is_uv);
    }
    SmartPtr<CLImage> get_ouput () {
        return _blender->get_blend_image (_layer, _is_uv);
    }
    const Rect &get_merge_window () {
        return _blender->_pyramid_layers[_layer].merge_window;
    }
    XCAM_DEAD_COPY (CLLinearBlenderKernel);

private:
    SmartPtr<CLPyramidBlender>    _blender;
    bool                          _is_uv;
    uint32_t                      _layer;
    Rect                          _merge_window;
    uint32_t                      _blend_width;
    uint32_t                      _input_width[XCAM_CL_BLENDER_IMAGE_NUM];
};

class CLPyramidTransformKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidTransformKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, uint32_t buf_index, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    SmartPtr<CLImage> get_input_gauss () {
        return _blender->get_gauss_image (_layer, _buf_index, _is_uv);
    }
    SmartPtr<CLImage> get_output_gauss () {
        // need reset format
        return _blender->get_gauss_image (_layer + 1, _buf_index, _is_uv);
    }
    SmartPtr<CLImage> get_output_lap () {
        return _blender->get_lap_image (_layer, _buf_index, _is_uv);
    }

    XCAM_DEAD_COPY (CLPyramidTransformKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    uint32_t                           _layer;
    uint32_t                           _buf_index;
    bool                               _is_uv;
    SmartPtr<CLImage>                  _output_gauss;
};

class CLPyramidReconstructKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidReconstructKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
private:
    SmartPtr<CLImage>  get_input_gauss() {
        return _blender->get_reconstruct_image (_layer + 1, _is_uv);
    }
    SmartPtr<CLImage>  get_input_lap() {
        return _blender->get_blend_image (_layer, _is_uv);
    }
    SmartPtr<CLImage>  get_output_gauss() {
        return _blender->get_reconstruct_image (_layer, _is_uv);
    }

    XCAM_DEAD_COPY (CLPyramidReconstructKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    uint32_t                           _layer;
    bool                               _is_uv;
    SmartPtr<CLImage>                  _input_gauss;
    int                                _out_gauss_width;
    int                                _out_gauss_height;
};

SmartPtr<CLImageHandler>
create_pyramid_blender (SmartPtr<CLContext> &context, int layer = 1, bool need_uv = true);

};

#endif //XCAM_CL_PYRAMID_BLENDER_H

