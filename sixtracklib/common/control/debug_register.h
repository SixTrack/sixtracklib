#ifndef SIXTRACKLIB_COMMON_CONTROL_DEBUGGING_REGISTER_H__
#define SIXTRACKLIB_COMMON_CONTROL_DEBUGGING_REGISTER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* #if !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( __cplusplus ) && defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

SIXTRL_STATIC SIXTRL_FN NS(arch_debugging_t) NS(DebugReg_store_arch_status)(
    NS(arch_debugging_t debugging_register,
    NS(arch_status_t) const status_flags_to_add );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(DebugReg_get_stored_arch_status)(
    NS(arch_debugging_t) const debugging_register );

SIXTRL_STATIC SIXTRL_FN NS(arch_debugging_t) NS(DebugReg_reset_arch_status)(
    NS(arch_debugging_t) const debugging_register );

SIXTRL_STATIC SIXTRL_FN bool NS(DebugReg_has_any_flags_set)(
    NS(arch_debugging_t) const debugging_register );

SIXTRL_STATIC SIXTRL_FN bool NS(DebugReg_has_status_flags_set)(
    NS(arch_debugging_t) const debugging_register );

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC SIXTRL_FN NS(arch_debugging_t)
NS(DebugReg_get_highest_set_flag)( NS(arch_debugging_t) debugging_register );

SIXTRL_STATIC SIXTRL_FN NS(arch_debugging_t)
NS(DebugReg_get_next_error_flag)(
    NS(arch_debugging_t) const debugging_register );

SIXTRL_STATIC SIXTRL_FN NS(arch_debugging_t)
NS(DebugReg_raise_next_error_flag)( NS(arch_debugging_t) debugging_register );

#if !defined( __cplusplus ) && defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}
#endif /* C++, Host */

/* ------------------------------------------------------------------------- */
/* -----          inline/header-only function implementation        -------- */
/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_debugging_t) NS(DebugReg_store_arch_status)(
    NS(arch_debugging_t debugging_register,
    NS(arch_status_t) const status_flags_to_add )
{
    NS(arch_debugging_t) flags_bitmask = ( arch_debugging_t)0u;

    if( status_flags_to_add >= ( arch_status_t )0 )
    {
        flags_bitmask = ( arch_debugging_t)status_flags_to_add;
    }
    else
    {
        arch_debugging_t flags_bitmask = -status_flags_to_add;
        flags_bitmask = ~( flags_bitmask );
        flags_bitmask += ( arch_debugging_t )1u;
    }

    debugging_register &= ~( SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK );
    return debugging_register | flags_bitmask;
}

SIXTRL_INLINE NS(arch_status_t) NS(DebugReg_get_stored_arch_status)(
    NS(arch_debugging_t) const debugging_register )
{
    NS(arch_status_t) restored_status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    NS(arch_debugging_t) flags_bitmask =
        debugging_register & SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK;

    if( flags_bitmask >= SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG )
    {
        flags_bitmask -= ( NS(arch_debugging_t) )1u;
        flags_bitmask  = ~flags_bitmask;
    }

    SIXTRL_ASSERT( ( flags_bitmask & SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG ) !=
        SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG );

    restored_status = ( NS(arch_status_t) )flags_bitmask;
    return restored_status;
}

SIXTRL_INLINE NS(arch_debugging_t) NS(DebugReg_reset_arch_status)(
    NS(arch_debugging_t) const debugging_register )
{
    return debugging_register & ~( SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK );
}

SIXTRL_INLINE bool NS(DebugReg_has_any_flags_set)(
    NS(arch_debugging_t) const debugging_register )
{
    return ( debugging_register != SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY );
}

SIXTRL_INLINE bool NS(DebugReg_has_status_flags_set)(
    NS(arch_debugging_t) const debugging_register )
{
    return ( ( debugging_register & SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK ) !=
        ( NS(arch_debugging_t) )SIXTRL_ARCH_STATUS_SUCCESS );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_debugging_t) NS(DebugReg_get_highest_set_flag)(
    NS(arch_debugging_t) debugging_register )
{
    /* Find the highest order bit in debugging_registers;
     * cf. https://stackoverflow.com/a/53184 */

    debugging_register |= ( debugging_register >>  1 );
    debugging_register |= ( debugging_register >>  2 );
    debugging_register |= ( debugging_register >>  4 );
    debugging_register |= ( debugging_register >>  8 );
    debugging_register |= ( debugging_register >> 16 );
    debugging_register |= ( debugging_register >> 32 );

    return debuging_register ^ ( debugging_register >> 1 );
}

SIXTRL_INLINE NS(arch_debugging_t) NS(DebugReg_get_next_error_flag)(
    NS(arch_debugging_t) const debugging_register )
{
    NS(arch_debugging_t) const highest_flag =
        NS(DebugReg_get_highest_set_flag)( debugging_register );

    if( highest_flag < SIXTRL_ARCH_DEBUGGING_MIN_FLAG )
    {
        return SIXTRL_ARCH_DEBUGGING_MIN_FLAG;
    }
    else if( highest_flag < SIXTRL_ARCH_DEBUGGING_MAX_FLAG )
    {
        return highest_flag + ( NS(arch_debugging_t) )1u;
    }

    return ( NS(arch_debugging_t) )0u;
}

SIXTRL_INLINE NS(arch_debugging_t) NS(DebugReg_raise_next_error_flag)(
    NS(arch_debugging_t) debugging_register )
{
    debugging_register |= NS(DebugReg_get_next_error_flag)(debugging_register);
    return debugging_register;
}

#endif /* SIXTRACKLIB_COMMON_CONTROL_DEBUGGING_REGISTER_H__ */

/* end: sixtracklib/common/control/debug_register.h */