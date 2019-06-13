#ifndef SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/internal/argument_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_has_cuda_arg_buffer)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(cuda_arg_buffer_t)
NS(CudaArgument_get_cuda_arg_buffer)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg );

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__ */
/* end: sixtracklib/cuda/internal/argument_base.h */
