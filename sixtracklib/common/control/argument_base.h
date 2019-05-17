#ifndef SIXTRACKLIB_COMMON_CONTROL_ARGUMENT_BASE_H__
#define SIXTRACKLIB_COMMON_CONTROL_ARGUMENT_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/argument_base.hpp"
    #include "sixtracklib/common/control/controller_base.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++ */

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Argument_delete)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_id_t) NS(Argument_get_arch_id)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_has_arch_string)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN char const* NS(Argument_get_arch_string)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );



SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(Argument_send_again)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(Argument_send_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    const NS(Buffer) *const SIXTRL_RESTRICT_REF buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Argument_send_buffer_without_remap)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    const NS(Buffer) *const SIXTRL_RESTRICT_REF buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(Argument_send_raw_argument)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    void const* SIXTRL_RESTRICT arg_begin,
    NS(arch_size_t) const arg_size );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Argument_receive_again)( NS(ArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Argument_receive_buffer)( NS(ArgumentBase)* SIXTRL_RESTRICT arg,
                             NS(Buffer)* SIXTRL_RESTRICT buf );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Argument_receive_buffer_without_remap)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    NS(Buffer)* SIXTRL_RESTRICT buf );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Argument_receive_raw_argument)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg,
    void* SIXTRL_RESTRICT arg_begin,
    NS(arch_size_t) const arg_capacity );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(Argument_remap_cobjects_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );



SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_uses_cobjects_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer) const*
NS(Argument_get_const_cobjects_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(Argument_get_cobjects_buffer)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t)
NS(Argument_get_cobjects_buffer_slot_size)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_uses_raw_argument)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN void const*
NS(Argument_get_const_ptr_raw_argument)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN void* NS(Argument_get_ptr_raw_argument)(
    NS(ArgumentBase)* SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t) NS(Argument_get_size)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_size_t) NS(Argument_get_capacity)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_has_argument_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(Argument_requires_argument_buffer)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );


SIXTRL_EXTERN SIXTRL_HOST_FN NS(ControllerBase)*
NS(Argument_get_ptr_base_controller)( NS(ArgumentBase)* SIXTRL_RESTRICT arg );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ControllerBase) const*
NS(Argument_get_const_ptr_base_controller)(
    const NS(ArgumentBase) *const SIXTRL_RESTRICT arg );

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++ */

#endif /* SIXTRACKLIB_COMMON_CONTROL_ARGUMENT_BASE_H__ */
/* end: sixtracklib/common/control/argument_base.h */
