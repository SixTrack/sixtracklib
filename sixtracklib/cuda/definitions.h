#ifndef SIXTRACKLIB_CUDA_DEFINITIONS_H__
#define SIXTRACKLIB_CUDA_DEFINITIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

typedef void* NS(cuda_arg_buffer_t);
typedef void const* NS(cuda_const_arg_buffer_t);

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

namespace SIXTRL_CXX_NAMESPACE
{
    typedef ::NS(cuda_arg_buffer_t)       cuda_arg_buffer_t;
    typedef ::NS(cuda_const_arg_buffer_t) cuda_const_arg_buffer_t;
}

#endif /* C++, Host */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_DEFINITIONS_H__ */

/* end: sixtracklib/cuda/definitions.h */
