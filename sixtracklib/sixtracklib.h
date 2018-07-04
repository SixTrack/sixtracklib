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
#include "sixtracklib/common/blocks.h"

#include "sixtracklib/common/impl/beam_elements_type.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/impl/beam_elements_api.h"

#include "sixtracklib/common/impl/particles_type.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/impl/particles_api.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_SIMD == 1 )

    #include "sixtracklib/simd/track.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_SIMD ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

    #include "sixtracklib/opencl/ocl_environment.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
           ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

    #include "sixtracklib/cuda/cuda_env.h"

#endif /* defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) */


/* ------------------------------------------------------------------------- */

#if defined( SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE )
    #include "sixtracklib/_impl/namespace_end.h"

    #undef __NAMESPACE
    #undef __SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE
#endif /* defined( SIXTRACKLIB_SIXTRACKLIB_UNDEF_NAMESPACE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_H__ */

/* end: sixtracklib/sixtracklib.h */
