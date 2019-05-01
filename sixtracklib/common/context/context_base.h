#ifndef SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__
#define SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/context/definitions.h"
    #include "sixtracklib/common/context/context_base.hpp"
    #include "sixtracklib/common/context/argument_base.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, host */

#if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Context_delete)(
    NS(ContextBase)* SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Context_clear)(
    NS(ContextBase)* SIXTRL_RESTRICT ctx );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(Context_get_arch_id)(
    const NS(ContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Context_has_arch_string)(
    const NS(ContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(Context_get_arch_string)(
    const NS(ContextBase) *const SIXTRL_RESTRICT ctx );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Context_has_config_string)(
    const NS(ContextBase) *const SIXTRL_RESTRICT ctx );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(Context_get_config_string)(
    const NS(ContextBase) *const SIXTRL_RESTRICT ctx );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Context_uses_nodes)(
    const NS(ContextBase) *const SIXTRL_RESTRICT ctx );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Context_send_detailed)(
    NS(ContextBase)* SIXTRL_RESTRICT ctx,
    NS(ArgumentBase)* SIXTRL_RESTRICT destination,
    void const* SIXTRL_RESTRICT source, NS(context_size_t) const src_len );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Context_send_buffer)(
    NS(ContextBase)* SIXTRL_RESTRICT ctx,
    NS(ArgumentBase)* SIXTRL_RESTRICT destination,
    NS(Buffer) const* SIXTRL_RESTRICT source );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Context_receive_detailed)(
    NS(ContextBase)* SIXTRL_RESTRICT ctx,
    void* SIXTRL_RESTRICT destination,
    NS(context_size_t) const destination_capacity,
    NS(ArgumentBase)* SIXTRL_RESTRICT source );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t) NS(Context_receive_buffer)(
    NS(ContextBase)* SIXTRL_RESTRICT ctx,
    NS(Buffer)* SIXTRL_RESTRICT destination,
    NS(ArgumentBase)* SIXTRL_RESTRICT source );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_status_t)
NS(Context_remap_sent_cobjects_buffer)(
    NS(ContextBase)* SIXTRL_RESTRICT ctx,
    NS(ArgumentBase)* SIXTRL_RESTRICT source );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Context_is_ready_to_remap)(
    const NS(ContextBase) *const SIXTRL_RESTRICT context );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Context_is_ready_to_send)(
    const NS(ContextBase) *const SIXTRL_RESTRICT context );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Context_is_ready_to_receive)(
    const NS(ContextBase) *const SIXTRL_RESTRICT context );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Context_is_in_debug_mode)(
    const NS(ContextBase) *const SIXTRL_RESTRICT context );

#endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, host */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_BASE_H__ */

/* end: sixtracklib/common/context/context_base.h */
