#ifndef SIXTRACKLIB_SIXTRACKLIB_HPP__
#define SIXTRACKLIB_SIXTRACKLIB_HPP__

/* ------------------------------------------------------------------------- */

#include "sixtracklib/common/generated/namespace.h"
#include "sixtracklib/common/generated/config.h"
#include "sixtracklib/common/definitions.h"

#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/be_drift/be_drift.hpp"
#include "sixtracklib/common/be_monitor/be_monitor.hpp"
#include "sixtracklib/common/output/output_buffer.hpp"
#include "sixtracklib/common/particles.hpp"

#include "sixtracklib/sixtracklib.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_SIMD == 1 )

//     #include "sixtracklib/simd/track.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

//     #include "sixtracklib/opencl/argument.h"
//     #include "sixtracklib/opencl/context.h"

//     #include "sixtracklib/opencl/buffer.h"
//     #include "sixtracklib/opencl/ocl_environment.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

//     #include "sixtracklib/cuda/buffer.h"
//     #include "sixtracklib/cuda/impl/track_particles_kernel_c_wrapper.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */


/* ------------------------------------------------------------------------- */

#endif /* SIXTRACKLIB_SIXTRACKLIB_HPP__ */

/* end: sixtracklib/sixtracklib.hpp */
