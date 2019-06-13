#ifndef SIXTRACKLIB_CUDA_ARGUMENT_H__
#define SIXTRACKLIB_CUDA_ARGUMENT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/argument.hpp"
    #include "sixtracklib/cuda/controller.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)* NS(CudaArgument_new)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaArgument_new_from_buffer)( NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(CudaController)* SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaArgument_new_from_size)( NS(ctrl_size_t) const capacity,
    NS(CudaController)* SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaArgument_new_from_raw_argument)(
    void const* SIXTRL_RESTRICT raw_arg_begin,
    NS(ctrl_size_t) const arg_length, NS(CudaController)* SIXTRL_RESTRICT c );

/* ------------------------------------------------------------------------- */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_has_cuda_arg_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(cuda_arg_buffer_t)
NS(CudaArgument_get_cuda_arg_buffer)( NS(CudaArgument)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(cuda_const_arg_buffer_t)
NS(CudaArgument_get_const_cuda_arg_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT arg );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
NS(CudaArgument_get_cuda_arg_buffer_as_cobject_buffer_begin)(
    NS(CudaArgument)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_cobject_buffer_begin)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT arg );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(arch_debugging_t)*
NS(CudaArgument_get_cuda_arg_buffer_as_debugging_register_begin)(
    NS(CudaArgument)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(arch_debugging_t) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_debugging_register_begin)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT arg );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
NS(CudaArgument_get_cuda_arg_buffer_as_elem_by_elem_config_begin)(
    NS(CudaArgument)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig) const*
NS(CudaArgument_get_cuda_arg_buffer_as_const_elem_by_elem_config_begin)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT arg );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_CUDA_ARGUMENT_H__ */

/* end: sixtracklib/cuda/argument.h */
