#ifndef SIXTRACKLIB_CUDA_DEFINITIONS_H__
#define SIXTRACKLIB_CUDA_DEFINITIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef void* NS(cuda_arg_buffer_t);

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    using cuda_arg_buffer_t = ::NS(cuda_arg_buffer_t);
}

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_CUDA_DEFINITIONS_H__ */

/* end: sixtracklib/cuda/definitions.h */
