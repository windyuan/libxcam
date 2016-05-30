/*
 * test-image-blend.cpp - test cl image
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

#include "test_common.h"
#include "cl_device.h"
#include "cl_context.h"
#include "cl_context.h"
#include "cl_pyramid_blender.h"


using namespace XCam;

class ImageFileHandle {
public:

    ImageFileHandle ()
        : _fp (NULL)
        , _file_name (NULL)
    {}
    ~ImageFileHandle ()
    {
        if (_fp)
            fclose (_fp);
        if (_file_name)
            xcam_free (_file_name);
    }

    bool is_valid () const {
        return (_fp ? true : false);
    }
    bool endOfFile();
    XCamReturn open (const char *name, const char *option);
    XCamReturn read_buf (SmartPtr<BufferProxy> &buf);
    XCamReturn write_buf (SmartPtr<BufferProxy> buf);

private:
    FILE   *_fp;
    char   *_file_name;
};

XCamReturn
ImageFileHandle::open (const char *name, const char *option)
{
    XCAM_ASSERT (name);
    if (!name)
        return XCAM_RETURN_ERROR_FILE;
    XCAM_ASSERT (!_file_name && !_fp);
    _fp = fopen (name, option);

    if (!_fp)
        return XCAM_RETURN_ERROR_FILE;
    _file_name = strndup (name, 512);
    return XCAM_RETURN_NO_ERROR;
}

bool
ImageFileHandle::endOfFile()
{
    if (!is_valid ())
        return true; // maybe false?

    return feof (_fp);
}

XCamReturn
ImageFileHandle::read_buf (SmartPtr<BufferProxy> &buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (is_valid ());

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fread (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, _fp) != line_bytes) {
                if (endOfFile ())
                    ret = XCAM_RETURN_BYPASS;
                else {
                    XCAM_LOG_ERROR ("read file failed, size doesn't match");
                    ret = XCAM_RETURN_ERROR_FILE;
                }
            }
        }
    }
    buf->unmap ();
    return ret;
}

XCamReturn
ImageFileHandle::write_buf (SmartPtr<BufferProxy> buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (is_valid ());

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fwrite (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, _fp) != line_bytes) {
                XCAM_LOG_ERROR ("read file failed, size doesn't match");
                ret = XCAM_RETURN_ERROR_FILE;
            }
        }
    }
    buf->unmap ();
    return ret;
}

void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s input0 input1 output\n", arg0);
}

int main (int argc, char *argv[])
{
    uint32_t input_format = V4L2_PIX_FMT_NV12;
    //uint32_t output_format = V4L2_PIX_FMT_NV12;
    uint32_t input_width = 1280;
    uint32_t input_height = 960;
    uint32_t output_width = 1920;
    uint32_t output_height = 960;

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLImageHandler> image_handler;
    VideoBufferInfo input_buf_info;
    SmartPtr<CLContext> context;
    SmartPtr<DrmDisplay> display;
    SmartPtr<BufferPool> buf_pool;
    ImageFileHandle file_in0, file_in1, file_out;
    const char *file_in0_name, *file_in1_name, *file_out_name;
    SmartPtr<DrmBoBuffer> output_buf;
    SmartPtr<BufferProxy> read_buf;

    if (argc < 4) {
        usage (argv[0]);
        return -1;
    }

    input_buf_info.init (input_format, input_width, input_height);
    display = DrmDisplay::instance ();
    buf_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (buf_pool.ptr ());
    buf_pool->set_video_info (input_buf_info);
    if (!buf_pool->reserve (6)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    file_in0_name = argv[1];
    file_in1_name = argv[2];
    file_out_name = argv[3];

    context = CLDevice::instance ()->get_context ();
    image_handler = create_pyramid_blender (context, 1);
    SmartPtr<CLPyramidBlender> blender = image_handler.dynamic_cast_ptr<CLPyramidBlender> ();
    XCAM_ASSERT (blender.ptr ());
    blender->set_output_size (output_width, output_height);

    SmartPtr<DrmBoBuffer> input0 = buf_pool->get_buffer (buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
    SmartPtr<DrmBoBuffer> input1 = buf_pool->get_buffer (buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_ASSERT (input0.ptr () && input1.ptr ());
    input0->attach_buffer (input1);

    //

    ret = file_in0.open (file_in0_name, "rb");
    CHECK (ret, "open input file(%s) failed", file_in0_name);
    read_buf = input0;
    ret = file_in0.read_buf (read_buf);
    CHECK (ret, "read buffer0 from (%s) failed", file_in0_name);

    ret = file_in1.open (file_in1_name, "rb");
    CHECK (ret, "open input file(%s) failed", file_in1_name);
    read_buf = input1;
    ret = file_in1.read_buf (read_buf);
    CHECK (ret, "read buffer1 from (%s) failed", file_in1_name);

    image_handler->execute (input0, output_buf);

    ret = file_out.open (file_out_name, "wb");
    CHECK (ret, "open output file(%s) failed", file_out_name);
    ret = file_out.write_buf (output_buf);
    CHECK (ret, "write buffer1 to (%s) failed", file_out_name);

    return ret;
}

