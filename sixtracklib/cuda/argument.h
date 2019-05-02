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
    #include "sixtracklib/cuda/internal/argument_base.h"
    #include "sixtracklib/cuda/argument.hpp"
    #include "sixtracklib/cuda/controller.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)* NS(CudaArgument_new)(
    NS(CudaController)* SIXTRL_RESTRICT ctrl );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaArgument_new_from_buffer)(
    NS(Buffer)* SIXTRL_RESTRICT buffer, NS(CudaController)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaArgument_new_from_size)(
    NS(controller_size_t) const capacity, NS(CudaController)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(CudaArgument)*
NS(CudaArgument_new_from_memory)(
    void const* SIXTRL_RESTRICT raw_arg_begin,
    NS(controller_size_t) const raw_arg_length,
    NS(CudaController)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(CudaArgument_delete)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(controller_status_t) NS(CudaArgument_send_buffer)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument,
    const NS(Buffer) *const SIXTRL_RESTRICT source_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(controller_status_t)
NS(CudaArgument_send_memory)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument,
    const void *const SIXTRL_RESTRICT source_arg_begin,
    NS(controller_size_t) const source_arg_length );

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(controller_status_t) NS(CudaArgument_receive_buffer)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument,
    NS(Buffer)* SIXTRL_RESTRICT destination_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN
NS(controller_status_t) NS(CudaArgument_receive_memory)(
    NS(CudaArgument)* SIXTRL_RESTRICT argument,
    void* SIXTRL_RESTRICT destination_buffer,
    NS(controller_size_t) const destination_capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_uses_cobjects_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(CudaArgument_get_cobjects_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_uses_raw_argument)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_get_ptr_raw_argument)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(controller_size_t) NS(CudaArgument_get_size)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(controller_size_t) NS(CudaArgument_get_capacity)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_has_argument_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(CudaArgument_requires_argument_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(CudaArgument_get_arch_id)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(CudaArgument_get_arch_string)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument );

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

#endif /* SIXTRACKLIB_CUDA_ARGUMENT_H__ */

/* end: sixtracklib/cuda/argument.h */
