#ifndef SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__
#define SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/control/argument_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_has_cuda_arg_buffer)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(cuda_arg_buffer_t)
NS(CudaArgument_get_cuda_arg_buffer)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(cuda_const_arg_buffer_t)
NS(CudaArgument_get_const_cuda_arg_buffer)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(arch_debugging_t)*
NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(arch_debugging_t) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
NS(CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin)(
    NS(CudaArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin)(
    const NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_CUDA_INTERNAL_ARGUMENT_BASE_H__ */
/* end: sixtracklib/cuda/control/argument_base.h */
