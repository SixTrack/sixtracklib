#ifndef SIXTRACKLIB_SIXTRACKLIB_H__
#define SIXTRACKLIB_SIXTRACKLIB_H__

#if !defined( __NAMESPACE )
    #define __NAMESPACE st_
    #define __SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE 1
#endif /* !defined( __NAMESPACE ) */

/* ------------------------------------------------------------------------- */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/_impl/namespace_begin.h"
#include "sixtracklib/_impl/path.h"
#include "sixtracklib/_impl/modules.h"

#include "sixtracklib/common/mem_pool.h"
#include "sixtracklib/common/compute_arch.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/impl/buffer_type.h"
#include "sixtracklib/common/impl/buffer_object.h"
#include "sixtracklib/common/impl/buffer_generic.h"
#include "sixtracklib/common/impl/managed_buffer.h"
#include "sixtracklib/common/impl/managed_buffer_minimal.h"
#include "sixtracklib/common/impl/managed_buffer_remap.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/track.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_SIMD == 1 )

    #include "sixtracklib/simd/track.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

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

#if defined( SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE )
    #include "sixtracklib/_impl/namespace_end.h"

    #undef __NAMESPACE
    #undef __SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE
#endif /* defined( SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_H__ */

/* end: sixtracklib/sixtracklib.h */
