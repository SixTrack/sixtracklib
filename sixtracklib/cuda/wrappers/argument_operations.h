#ifndef SIXTRACKLIB_CUDA_WRAPPERS_ARGUMENT_OPERATIONS_H__
#define SIXTRACKLIB_CUDA_WRAPPERS_ARGUMENT_OPERATIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/context/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_EXTERN void* NS(CudaArgument_alloc_arg_buffer)(
    NS(context_size_t) const capacity );

SIXTRL_EXTERN void NS(CudaArgument_free_arg_buffer)(
    void* SIXTRL_RESTRICT arg_buffer );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_CUDA_WRAPPERS_ARGUMENT_OPERATIONS_H__ */

/* end: sixtracklib/cuda/wrappers/argument_operations.h */
