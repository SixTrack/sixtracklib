#include "sixtracklib/common/control/arch_info.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/testlib.h"

TEST( C99_CommonControlArchInfoTests, MinimalUsage )
{
    using arch_info_t = ::NS(ArchInfo);

    /* Create NS(ArchInfo) instance with inital values */

    arch_info_t* arch_info_a = ::NS(ArchInfo_create)();
    ASSERT_TRUE( arch_info_a != nullptr );

    /* Verify the initial state */

    ASSERT_TRUE( ::NS(ArchInfo_get_arch_id)( arch_info_a ) ==
                 ::NS(ARCHITECTURE_ILLEGAL) );

    ASSERT_TRUE( !::NS(ArchInfo_has_arch_string)( arch_info_a ) );
    ASSERT_TRUE(  ::NS(ArchInfo_get_arch_string)( arch_info_a ) != nullptr );
    ASSERT_TRUE(  std::strlen( ::NS(ArchInfo_get_arch_string)(
                    arch_info_a ) ) == std::size_t{ 0 } );

    /* Reset the arch info to represent a member of the CPU architecture */

    NS(ArchInfo_reset)( arch_info_a, ::NS(ARCHITECTURE_CPU),
                        SIXTRL_ARCHITECTURE_CPU_STR );

    /* Verify that the values are now consistent with this choice */

    ASSERT_TRUE( ::NS(ArchInfo_get_arch_id)( arch_info_a ) ==
                 ::NS(ARCHITECTURE_CPU) );

    ASSERT_TRUE( ::NS(ArchInfo_has_arch_string)( arch_info_a ) );
    ASSERT_TRUE( ::NS(ArchInfo_get_arch_string)( arch_info_a ) != nullptr );
    ASSERT_TRUE( 0 == std::strcmp(
        ::NS(ArchInfo_get_arch_string)( arch_info_a ),
        SIXTRL_ARCHITECTURE_CPU_STR ) );

    /* Create a second ArchInfo instance, also from the CPU architecture */

    arch_info_t* arch_info_b = ::NS(ArchInfo_new)(
        ::NS(ARCHITECTURE_CPU), SIXTRL_ARCHITECTURE_CPU_STR );

    ASSERT_TRUE( arch_info_b != nullptr );

    /* Verify that also arch_info_b is consistent with this choice */

    ASSERT_TRUE( ::NS(ArchInfo_get_arch_id)( arch_info_b ) ==
                 ::NS(ARCHITECTURE_CPU) );

    ASSERT_TRUE( ::NS(ArchInfo_has_arch_string)( arch_info_b ) );
    ASSERT_TRUE( ::NS(ArchInfo_get_arch_string)( arch_info_b ) != nullptr );
    ASSERT_TRUE( 0 == std::strcmp(
        ::NS(ArchInfo_get_arch_string)( arch_info_a ),
        ::NS(ArchInfo_get_arch_string)( arch_info_b ) ) );

    /* verify that arch_info_a and arch_info_b compare as identical
     * to each other */

    ASSERT_TRUE( ::NS(ArchInfo_is_identical_to)( arch_info_a, arch_info_b ) );
    ASSERT_TRUE( ::NS(ArchInfo_is_identical_to_arch_id)( arch_info_a,
                 ::NS(ArchInfo_get_arch_id)( arch_info_b ) ) );

    ASSERT_TRUE( ::NS(ArchInfo_is_compatible_with)(
        arch_info_a, arch_info_b ) );

    ASSERT_TRUE( ::NS(ArchInfo_is_compatible_with_arch_id)( arch_info_a,
                 ::NS(ArchInfo_get_arch_id)( arch_info_b ) ) );

    /* reset arch_info_a to represent now an OpenCL architecture */

    ::NS(ArchInfo_reset)( arch_info_a, ::NS(ARCHITECTURE_OPENCL),
                          SIXTRL_ARCHITECTURE_OPENCL_STR );

    ASSERT_TRUE( ::NS(ArchInfo_get_arch_id)( arch_info_a ) ==
                 ::NS(ARCHITECTURE_OPENCL) );

    ASSERT_TRUE( ::NS(ArchInfo_has_arch_string)( arch_info_a ) );
    ASSERT_TRUE( ::NS(ArchInfo_get_arch_string)( arch_info_a ) != nullptr );
    ASSERT_TRUE( 0 == std::strcmp(
        ::NS(ArchInfo_get_arch_string)( arch_info_a ),
        SIXTRL_ARCHITECTURE_OPENCL_STR ) );

    /* arch_info_a and arch_info_b should no longer be considered equal /
     * compatible with each other */

    ASSERT_TRUE( !::NS(ArchInfo_is_identical_to)( arch_info_a, arch_info_b ) );
    ASSERT_TRUE( !::NS(ArchInfo_is_identical_to_arch_id)( arch_info_a,
                  ::NS(ArchInfo_get_arch_id)( arch_info_b ) ) );

    ASSERT_TRUE( !::NS(ArchInfo_is_compatible_with)(
        arch_info_a, arch_info_b ) );

    ASSERT_TRUE( !::NS(ArchInfo_is_compatible_with_arch_id)( arch_info_a,
                  ::NS(ArchInfo_get_arch_id)( arch_info_b ) ) );

    /* Clean-up */

    ::NS(ArchInfo_delete)( arch_info_a );
    arch_info_a = nullptr;

    ::NS(ArchInfo_delete)( arch_info_b );
    arch_info_b = nullptr;
}

/* end: tests/sixtracklib/common/control/test_arch_info_c99.cpp */
