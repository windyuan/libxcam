/*
 * cl_3a_stats_context.h - CL 3a stats context
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

#ifndef XCAM_CL_3A_STATS_CONTEXT_H
#define XCAM_CL_3A_STATS_CONTEXT_H

#include "xcam_utils.h"
#include "cl_memory.h"
#include "x3a_stats_pool.h"
#include "cl_context.h"

#define XCAM_CL_3A_STATS_BUFFER_COUNT 6

namespace XCam {

class CL3AStatsCalculatorContext
{
public:
    CL3AStatsCalculatorContext (const SmartPtr<CLContext> &context);
    ~CL3AStatsCalculatorContext ();

    bool is_ready () const {
        return _data_allocated;
    }
    bool allocate_data (const VideoBufferInfo &buffer_info);
    void pre_stop ();
    void clean_up_data ();

    SmartPtr<CLBuffer> get_next_buffer ();
    SmartPtr<X3aStats> copy_stats_out (const SmartPtr<CLBuffer> &stats_cl_buf);

private:
    XCAM_DEAD_COPY (CL3AStatsCalculatorContext);

    bool fill_histogram (XCam3AStats *stats);

private:
    SmartPtr<CLContext>              _context;
    SmartPtr<X3aStatsPool>           _stats_pool;
    SmartPtr<CLBuffer>               _stats_cl_buffer[XCAM_CL_3A_STATS_BUFFER_COUNT];
    uint32_t                         _stats_buf_index;
    XCam3AStatsInfo                  _stats_info;
    bool                             _data_allocated;
};

}
#endif //XCAM_CL_3A_STATS_CONTEXT_H
