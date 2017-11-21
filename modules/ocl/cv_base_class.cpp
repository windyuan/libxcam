/*
 * cv_base_class.cpp - base class for all OpenCV related features
 *
 *  Copyright (c) 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Andrey Parfenov <a1994ndrey@gmail.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "cv_base_class.h"

namespace XCam {

CVBaseClass::CVBaseClass ()
{
    _cv_context = CVContext::instance ();
    XCAM_ASSERT (_cv_context.ptr ());
    _use_ocl = cv::ocl::useOpenCL ();
}

bool
CVBaseClass::convert_to_mat (SmartPtr<VideoBuffer> buffer, cv::Mat &image)
{
    cv::Mat mat;
    VideoBufferInfo info = buffer->get_video_info ();
    XCAM_FAIL_RETURN (WARNING, info.format == V4L2_PIX_FMT_NV12, false, "convert_to_mat only support NV12 format");

    SmartPtr<CLBuffer> cl_buffer = convert_to_clbuffer (_cv_context->get_cl_context (), buffer);
    if (!cl_buffer.ptr ()) {
        uint8_t *ptr = buffer->map ();
        mat = cv::Mat (info.height * 3 / 2, info.width, CV_8UC1, ptr, info.strides[0]);
    } else {
        cl_mem cl_mem_id = cl_buffer->get_mem_id ();

        cv::UMat umat;
        cv::ocl::convertFromBuffer (cl_mem_id, info.strides[0], info.height * 3 / 2, info.width, CV_8U, umat);
        if (umat.empty ()) {
            XCAM_LOG_ERROR ("convert buffer to UMat failed");
            return false;
        }

        umat.copyTo (mat);
        if (mat.empty ()) {
            XCAM_LOG_ERROR ("copy UMat to Mat failed");
            return false;
        }
    }

    cv::cvtColor (mat, image, cv::COLOR_YUV2BGR_NV12);
    return true;
}

}
