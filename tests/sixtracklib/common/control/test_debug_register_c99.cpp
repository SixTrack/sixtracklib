#include "sixtracklib/common/control/debug_register.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/testlib.h"

TEST( C99_CommonControlDebuggingRegisterTests, BasicUsage )
{
    using arch_debugging_t = ::NS(arch_debugging_t);
    using arch_status_t    = ::NS(arch_status_t);

    /* Init empty debug_register */

    arch_debugging_t debug_register = ::NS(ARCH_DEBUGGING_REGISTER_EMPTY);

    ASSERT_TRUE( ::NS(ARCH_DEBUGGING_REGISTER_EMPTY) ==
                 ::NS(DebugReg_get_highest_set_flag)( debug_register ) );

    ASSERT_TRUE( !::NS(DebugReg_has_any_flags_set)( debug_register ) );
    ASSERT_TRUE( !::NS(DebugReg_has_status_flags_set)( debug_register ) );

    /* Find next error flag; since the register is empty, this should be
     * NS(ARCH_DEBUGGING_MIN_FLAG) */

    arch_debugging_t next_flag =
        ::NS(DebugReg_get_next_error_flag)( debug_register );

    ASSERT_TRUE( next_flag == ::NS(ARCH_DEBUGGING_MIN_FLAG) )

    /* Apply the first error flag, and check whether the register is changed
     * accordingly */

    debug_register |= next_flag;

    ASSERT_TRUE(  ::NS(DebugReg_has_any_flags_set)( debug_register ) );
    ASSERT_TRUE( !::NS(DebugReg_has_status_flags_set)( debug_register ) );

    ASSERT_TRUE( next_flag == ::NS(DebugReg_get_highest_set_flag)(
        debug_register ) );

    /* Store the next highest error flag in the register */

    next_flag <<= 1; /* next higher manually */

    ASSERT_TRUE( next_flag == ::NS(DebugReg_get_next_error_flag)(
        debug_register ) );

    /* use automatic increment option to actually modify the register but ...*/
    ::NS(DebugReg_raise_next_error_flag)( debug_register );

    /* ... verify against the manually raised flag */
    ASSERT_TRUE( next_flag == ::NS(DebugReg_get_highest_set_flag)(
        debug_register ) );

    ASSERT_TRUE(  ::NS(DebugReg_has_any_flags_set)( debug_register ) );
    ASSERT_TRUE( !::NS(DebugReg_has_status_flags_set)( debug_register ) );

    /* Add second highest possible flag */

    next_flag = ::NS(ARCH_DEBUGGING_MAX_FLAG) >> 1u;

    debug_register |= next_flag;

    ASSERT_TRUE( ::NS(DebugReg_get_next_error_flag)( debug_register ) ==
                 ::NS(ARCH_DEBUGGING_MAX_FLAG) );

    ASSERT_TRUE(  ::NS(DebugReg_has_any_flags_set)( debug_register ) );
    ASSERT_TRUE( !::NS(DebugReg_has_status_flags_set)( debug_register ) );

    /* Add maximum possible flag to register */

    ::NS(DebugReg_raise_next_error_flag)( debug_register );

    /* Now it should not be possible to add any more flags */

    next_flag = ::NS(DebugReg_get_next_error_flag)( debug_register );

    ASSERT_TRUE( next_flag == ::NS(ARCH_DEBUGGING_REGISTER_EMPTY) );
    ASSERT_TRUE(  ::NS(DebugReg_has_any_flags_set)( debug_register ) );

    /* ********************************************************************* */
    /* Test handling of storing and retrieiving of NS(arch_status_t) flags   */

    arch_debugging_t const cpy_debug_register = debug_register;

    /* If no status flags are set, the restored status should evaluate to
     * NS(ARCH_STATUS_SUCCESS) */

    ASSERT_TRUE( !::NS(DebugReg_has_status_flags_set)( debug_register ) );
    arch_status_t const arch_status_ok = ::NS(DebugReg_get_stored_arch_status)(
        debug_register );

    ASSERT_TRUE( arch_status_ok == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( debug_register == cpy_debug_register );

    /* Store an empty arch_flag to the debug register -> should not change
     * anything */

    debug_register = ::NS(DebugReg_store_arch_status)(
        debug_register, arch_status_ok );

    ASSERT_TRUE( !::NS(DebugReg_has_status_flags_set)( debug_register ) );
    ASSERT_TRUE( debug_register == cpy_debug_register );

    /* storing a different status flag set will however change the
     * representation */

    arch_status_t const arch_status_fail = ::NS(ARCH_STATUS_GENERAL_FAILURE);

    debug_register = ::NS(DebugReg_store_arch_status)(
        debug_register, arch_status_fail );

    ASSERT_TRUE( ::NS(DebugReg_has_status_flags_set)( debug_register ) );
    ASSERT_TRUE( ::NS(DebugReg_has_any_flags_set)( debug_register ) );

    ASSERT_TRUE( cpy_debug_register ==
        ( debug_register & ( arch_debugging_t )arch_status_ok ) );

    ASSERT_TRUE( ::NS(DebugReg_get_stored_arch_status)( debug_register ) ==
                 arch_status_fail );

    /* Resetting the status flag part should turn everything back to its
     * former state */

    debug_register = NS(DebugReg_reset_arch_status)( debug_register );

    ASSERT_TRUE( cpy_debug_register != debug_register );
    ASSERT_TRUE( !::NS(DebugReg_has_status_flags_set)( debug_register ) );
}

/* end: tests/sixtracklib/common/control/test_debug_register_c99.cpp */
