#ifndef SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_H__
#define SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/context/argument_base.hpp"
    #include "sixtracklib/common/context/context_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, host */

#if defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_type_id_t) NS(Argument_get_type)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(Argument_get_ptr_type_strt)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );



SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Argument_send_again)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Argument_send_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    const NS(Buffer) *const SIXTRL_RESTRICT_REF buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Argument_send_memory)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    void const* SIXTRL_RESTRICT arg_begin, NS(context_size_t) const arg_size );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Argument_receive_again)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Argument_receive_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg, NS(Buffer)* SIXTRL_RESTRICT buf );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Argument_receive_memory)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg, void* SIXTRL_RESTRICT arg_begin,
    NS(context_size_t) const arg_capacity );


SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_uses_cobjects_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(Argument_get_const_cobjects_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(Argument_get_cobjects_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_is_using_raw_argument)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN void const*
NS(Argument_get_const_ptr_raw_argument)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN void* NS(Argument_get_ptr_raw_argument)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_size_t) NS(Argument_get_size)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_size_t) NS(Argument_get_capacity)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_has_argument_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_requires_argument_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(ContextBase) const*
NS(Argument_get_ptr_base_context)( NS(Argument)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ContextBase) const*
NS(Argument_get_const_ptr_base_context)(
    const NS(Argument) *const SIXTRL_RESTRICT arg );

#endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, host */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_ARGUMENT_BASE_H__ */
/* end: sixtracklib/common/context/argument_base.h */
