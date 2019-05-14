#ifndef SIXTRACKLIB_CUDA_ARGUMENT_H__
#define SIXTRACKLIB_CUDA_ARGUMENT_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/cuda/definitions.h"
    #include "sixtracklib/cuda/control/argument_base.h"
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
NS(CudaArgument_new_from_memory)( void const* SIXTRL_RESTRICT raw_arg_begin,
    NS(ctrl_size_t) const arg_length, NS(CudaController)* SIXTRL_RESTRICT c );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(CudaArgument_delete)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t) NS(CudaArgument_send_buffer)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument,
    const NS(Buffer) *const SIXTRL_RESTRICT source_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t)
NS(CudaArgument_send_memory)( NS(CudaArgument)* SIXTRL_RESTRICT argument,
    const void *const SIXTRL_RESTRICT source_arg_begin,
    NS(ctrl_size_t) const source_arg_length );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t) NS(CudaArgument_receive_buffer)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument,
    NS(Buffer)* SIXTRL_RESTRICT destination_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_status_t) NS(CudaArgument_receive_memory)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument,
    void* SIXTRL_RESTRICT dest_buffer, NS(ctrl_size_t) const dest_capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_uses_cobjects_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(CudaArgument_get_cobjects_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_uses_raw_argument)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_get_ptr_raw_argument)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t) NS(CudaArgument_get_size)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ctrl_size_t) NS(CudaArgument_get_capacity)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_has_argument_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_requires_argument_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(CudaArgument_get_arch_id)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(CudaArgument_get_arch_string)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_CUDA_ARGUMENT_H__ */

/* end: sixtracklib/cuda/argument.h */
