#ifndef SIXTRACKLIB_CUDA_CONTEXT_H__
#define SIXTRACKLIB_CUDA_CONTEXT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/context.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaContext)* NS(CudaContext_create)( void );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(CudaContext_delete)(
    NS(CudaContext)* SIXTRL_RESTRICT context );

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_CONTEXT_H__ */

/* end: sixtracklib/cuda/context.h */
