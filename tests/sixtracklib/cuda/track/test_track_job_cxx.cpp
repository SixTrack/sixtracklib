#include "sixtracklib/cuda/track_job.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/cuda/controller.hpp"
#include "sixtracklib/cuda/argument.hpp"

TEST( CXX_CudaTrackJobTests, BasicUsageCBuffer )
{
    namespace st = SIXTRL_CXX_NAMESPACE;
    using track_job_t = st::CudaTrackJob;
    using c_buffer_t    = track_job_t::buffer_t;

    c_buffer_t*


}

/* end: tests/sixtracklib/cuda/track/test_track_job_cxx.cpp */
