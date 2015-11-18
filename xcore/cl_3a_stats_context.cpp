/*
 * cl_3a_stats_context.cpp - CL 3a stats context
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
 * Author: Jia Meng <jia.meng@intel.com>
 */

#include "xcam_utils.h"
#include "cl_3a_stats_context.h"

namespace XCam {
CL3AStatsCalculatorContext::CL3AStatsCalculatorContext (const SmartPtr<CLContext> &context)
    : _context (context)
    , _stats_buf_index (0)
    , _data_allocated (false)
{
}

CL3AStatsCalculatorContext::~CL3AStatsCalculatorContext ()
{
}

bool
CL3AStatsCalculatorContext::allocate_data (const VideoBufferInfo &buffer_info)
{
    _stats_pool = new X3aStatsPool ();
    _stats_pool->set_video_info (buffer_info);

    XCAM_FAIL_RETURN (
        WARNING,
        _stats_pool->reserve (32), // need reserve more if as attachement
        false,
        "reserve cl stats buffer failed");

    _stats_info = _stats_pool->get_stats_info ();

    for (uint32_t i = 0; i < XCAM_CL_3A_STATS_BUFFER_COUNT; ++i) {
        _stats_cl_buffer[i] = new CLBuffer (
            _context,
            _stats_info.aligned_width * _stats_info.aligned_height * sizeof (XCamGridStat));

        XCAM_ASSERT (_stats_cl_buffer[i].ptr ());
        XCAM_FAIL_RETURN (
            WARNING,
            _stats_cl_buffer[i]->is_valid (),
            false,
            "allocate cl stats buffer failed");
    }
    _stats_buf_index = 0;
    _data_allocated = true;

    return true;
}

void
CL3AStatsCalculatorContext::pre_stop ()
{
    if (_stats_pool.ptr ())
        _stats_pool->stop ();
}


void
CL3AStatsCalculatorContext::clean_up_data ()
{
    _data_allocated = false;

    for (uint32_t i = 0; i < XCAM_CL_3A_STATS_BUFFER_COUNT; ++i) {
        _stats_cl_buffer[i].release ();
    }
    _stats_buf_index = 0;
}

SmartPtr<CLBuffer>
CL3AStatsCalculatorContext::get_next_buffer ()
{
    SmartPtr<CLBuffer> buf = _stats_cl_buffer[_stats_buf_index];
    _stats_buf_index = ((_stats_buf_index + 1) % XCAM_CL_3A_STATS_BUFFER_COUNT);
    return buf;
}

void debug_print_3a_stats (XCam3AStats *stats_ptr)
{

    for (int y = 30; y < 40; ++y) {
        printf ("---- y ");
        for (int x = 54; x < 64; ++x)
            printf ("%3d", stats_ptr->stats[y * stats_ptr->info.aligned_width + x].avg_y);
        printf ("\n");
    }

#if 0
#define DUMP_STATS(ch, w, h, aligned_w, stats) do {                 \
         printf ("stats " #ch ":");                                  \
         for (uint32_t y = 0; y < h; ++y) {                          \
             for (uint32_t x = 0; x < w; ++x)                        \
                 printf ("%3d ", stats[y * aligned_w + x].avg_##ch); \
         }                                                           \
         printf ("\n");                           \
     } while (0)
    DUMP_STATS (r,  stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (gr, stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (gb, stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (b,  stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (y,  stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
#endif
}

void debug_print_histogram (XCam3AStats *stats_ptr)
{
#define DUMP_HISTOGRAM(ch, bins, hist) do {      \
         printf ("histogram " #ch ":");           \
         for (uint32_t i = 0; i < bins; i++) {    \
             if (i % 16 == 0) printf ("\n");      \
             printf ("%4d ", hist[i].ch);         \
         }                                        \
         printf ("\n");                           \
     } while (0)

    DUMP_HISTOGRAM (r,  stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);
    DUMP_HISTOGRAM (gr, stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);
    DUMP_HISTOGRAM (gb, stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);
    DUMP_HISTOGRAM (b,  stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);

    printf ("histogram y:");
    for (uint32_t i = 0; i < stats_ptr->info.histogram_bins; i++) {
        if (i % 16 == 0) printf ("\n");
        printf ("%4d ", stats_ptr->hist_y[i]);
    }
    printf ("\n");
}

SmartPtr<X3aStats>
CL3AStatsCalculatorContext::copy_stats_out (const SmartPtr<CLBuffer> &stats_cl_buf)
{
    SmartPtr<BufferProxy> buffer;
    SmartPtr<X3aStats> stats;
    SmartPtr<CLEvent>  event = new CLEvent;
    XCam3AStats *stats_ptr = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (stats_cl_buf.ptr ());

    buffer = _stats_pool->get_buffer (_stats_pool);
    XCAM_FAIL_RETURN (WARNING, buffer.ptr (), NULL, "3a stats pool stopped.");

    stats = buffer.dynamic_cast_ptr<X3aStats> ();
    XCAM_ASSERT (stats.ptr ());
    stats_ptr = stats->get_stats ();
    ret = stats_cl_buf->enqueue_read (
              stats_ptr->stats,
              0, _stats_info.aligned_width * _stats_info.aligned_height * sizeof (stats_ptr->stats[0]),
              CLEvent::EmptyList, event);

    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats enqueue read buffer failed.");
    XCAM_ASSERT (event->get_event_id ());

    ret = event->wait ();
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats buffer enqueue event wait failed");
    event.release ();

    //debug_print_3a_stats (stats_ptr);
    fill_histogram (stats_ptr);
    //debug_print_histogram (stats_ptr);

    return stats;
}

bool
CL3AStatsCalculatorContext::fill_histogram (XCam3AStats * stats)
{
    const XCam3AStatsInfo &stats_info = stats->info;
    const XCamGridStat *grid_stat;
    XCamHistogram *hist_rgb = stats->hist_rgb;
    uint32_t *hist_y = stats->hist_y;

    memset (hist_rgb, 0, sizeof(XCamHistogram) * stats_info.histogram_bins);
    memset (hist_y, 0, sizeof(uint32_t) * stats_info.histogram_bins);
    for (uint32_t i = 0; i < stats_info.width; i++) {
        for (uint32_t j = 0; j < stats_info.height; j++) {
            grid_stat = &stats->stats[j * stats_info.aligned_width + i];
            hist_rgb[grid_stat->avg_r].r++;
            hist_rgb[grid_stat->avg_gr].gr++;
            hist_rgb[grid_stat->avg_gb].gb++;
            hist_rgb[grid_stat->avg_b].b++;
            hist_y[grid_stat->avg_y]++;
        }
    }
    return true;
}

}
