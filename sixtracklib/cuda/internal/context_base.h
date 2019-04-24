#ifndef SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/context/argument_base.hpp"
    #include "sixtracklib/common/context/context_base.hpp"
    #include "sixtracklib/common/context/context_base_with_nodes.hpp"
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

#if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(CudaContext_delete)(
    NS(CudaContextBase)* SIXTRL_RESTRICT context );

#endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_CONTEXT_BASE_H__ */

/* end: sixtracklib/cuda/internal/context_base.h */
