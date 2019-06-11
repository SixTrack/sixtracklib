#ifndef SIXTRACKLIB_CUDA_DEFINITIONS_H__
#define SIXTRACKLIB_CUDA_DEFINITIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

typedef void* NS(cuda_arg_buffer_t);
typedef void const* NS(cuda_const_arg_buffer_t);
typedef int NS(cuda_dev_index_t);

#if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

SIXTRL_STATIC_VAR NS(ctrl_size_t) const
    NS(ARCH_CUDA_DEFAULT_WARP_SIZE) = ( NS(ctrl_size_t) )32u;

#endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    typedef ::NS(cuda_arg_buffer_t)       cuda_arg_buffer_t;
    typedef ::NS(cuda_const_arg_buffer_t) cuda_const_arg_buffer_t;
    typedef ::NS(cuda_dev_index_t)        cuda_dev_index_t;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST ctrl_size_t
        ARCH_CUDA_DEFAULT_WARP_SIZE = static_cast< ctrl_size_t >( 32u );
}

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_DEFINITIONS_H__ */

/* end: sixtracklib/cuda/definitions.h */
