#ifndef SIXTRACKLIB_SIXTRACKLIB_HPP__
#define SIXTRACKLIB_SIXTRACKLIB_HPP__

/* ------------------------------------------------------------------------- */

#include "sixtracklib/common/generated/namespace.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/common/definitions.h"

#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/be_drift/be_drift.hpp"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"
#include "sixtracklib/common/be_limit/be_limit_rect.hpp"
#include "sixtracklib/common/be_limit/be_limit_ellipse.hpp"
#include "sixtracklib/common/be_dipedge/be_dipedge.hpp"
#include "sixtracklib/common/output/output_buffer.hpp"
#include "sixtracklib/common/particles.hpp"

#include "sixtracklib/sixtracklib.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_SIMD == 1 )

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

#include "sixtracklib/cuda/control/kernel_config.hpp"
#include "sixtracklib/cuda/control/node_info.hpp"

#include "sixtracklib/cuda/argument.hpp"
#include "sixtracklib/cuda/controller.hpp"
#include "sixtracklib/cuda/track_job.hpp"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */


/* ------------------------------------------------------------------------- */

#endif /* SIXTRACKLIB_SIXTRACKLIB_HPP__ */

/* end: sixtracklib/sixtracklib.hpp */

