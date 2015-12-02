/*
 * cl_bayer_horiz_nr_handler.h - CL bayer horizatal NR handler
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

#ifndef XCAM_CL_BAYER_HORIZ_HANLDER_H
#define XCAM_CL_BAYER_HORIZ_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

class CLBayerHorizNRImageKernel
    : public CLImageKernel
{
public:
    explicit CLBayerHorizNRImageKernel (SmartPtr<CLContext> &context, uint32_t index);

    virtual XCamReturn post_execute ();
protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLBayerHorizNRImageKernel);

    uint32_t   _component_index;
};

class CLBayerRotationImageKernel
    : public CLImageKernel
{
public:
    explicit CLBayerRotationImageKernel (SmartPtr<CLContext> &context, uint32_t index);

    virtual XCamReturn post_execute ();
protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLBayerRotationImageKernel);

    uint32_t   _component_index;
};


class CLRotationHandler
    : public CLImageHandler
{
    friend class CLBayerRotationImageKernel;

public:
    explicit CLRotationHandler (const char *name);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCAM_DEAD_COPY (CLRotationHandler);

};

SmartPtr<CLImageHandler>
create_cl_bayer_horiz_nr_image_handler (SmartPtr<CLContext> &context);

SmartPtr<CLImageHandler>
create_cl_bayer_rotation_image_handler (SmartPtr<CLContext> &context);


};

#endif //XCAM_CL_BAYER_HORIZ_HANLDER_H
