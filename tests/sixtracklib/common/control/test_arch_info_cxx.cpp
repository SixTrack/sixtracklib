#include "sixtracklib/common/control/arch_info.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/testlib.h"

TEST( CXX_CommonControlArchInfoTests, MinimalUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using arch_info_t = st::ArchInfo;

    /* Create st::ArchInfo instance with inital values */

    arch_info_t arch_info_a;

    /* Verify the initial state */

    ASSERT_TRUE( arch_info_a.archId() == st::ARCHITECTURE_ILLEGAL );
    ASSERT_TRUE( !arch_info_a.hasArchStr() );
    ASSERT_TRUE( arch_info_a.ptrArchStr() != nullptr );
    ASSERT_TRUE( std::strlen( arch_info_a.ptrArchStr() ) == std::size_t{ 0 } );

    std::string  arch_info_a_str = arch_info_a.archStr();
    ASSERT_TRUE( arch_info_a_str.empty() );

    /* Reset the arch info to represent a member of the CPU architecture */

    arch_info_a.reset( st::ARCHITECTURE_CPU, SIXTRL_ARCHITECTURE_CPU_STR );

    /* Verify that the values are now consistent with this choice */

    ASSERT_TRUE( arch_info_a.archId() == st::ARCHITECTURE_CPU );

    ASSERT_TRUE( arch_info_a.hasArchStr() );
    ASSERT_TRUE( arch_info_a.ptrArchStr() != nullptr );
    ASSERT_TRUE( 0 == std::strcmp( arch_info_a.ptrArchStr(),
                                   SIXTRL_ARCHITECTURE_CPU_STR ) );

    arch_info_a_str = arch_info_a.archStr();
    ASSERT_TRUE( arch_info_a_str.compare( SIXTRL_ARCHITECTURE_CPU_STR ) == 0 );

    /* Create a second ArchInfo instance, also from the CPU architecture */

    arch_info_t arch_info_b(st::ARCHITECTURE_CPU, SIXTRL_ARCHITECTURE_CPU_STR );

    /* Verify that also arch_info_b is consistent with this choice */

    ASSERT_TRUE( arch_info_b.archId() == st::ARCHITECTURE_CPU );

    ASSERT_TRUE( arch_info_b.hasArchStr() );
    ASSERT_TRUE( arch_info_b.ptrArchStr() != nullptr );
    ASSERT_TRUE( 0 == std::strcmp(
        arch_info_a.ptrArchStr(), arch_info_b.ptrArchStr() ) );

    /* verify that arch_info_a and arch_info_b compare as identical
     * to each other */

    ASSERT_TRUE( arch_info_a.isArchIdenticalTo( arch_info_b ) );
    ASSERT_TRUE( arch_info_a.isArchIdenticalTo( arch_info_b.archId() ) );

    ASSERT_TRUE( arch_info_a.isArchCompatibleWith( arch_info_b ) );
    ASSERT_TRUE( arch_info_a.isArchCompatibleWith( arch_info_b.archId() ) );

    /* reset arch_info_a to represent now an OpenCL architecture */

    arch_info_a.reset( st::ARCHITECTURE_OPENCL,
                       SIXTRL_ARCHITECTURE_OPENCL_STR );

    ASSERT_TRUE( arch_info_a.archId() == st::ARCHITECTURE_OPENCL );
    ASSERT_TRUE( arch_info_a.hasArchStr() );
    ASSERT_TRUE( arch_info_a.ptrArchStr() != nullptr );
    ASSERT_TRUE( 0 == std::strcmp( arch_info_a.ptrArchStr(),
        SIXTRL_ARCHITECTURE_OPENCL_STR ) );

    /* arch_info_a and arch_info_b should no longer be considered equal /
     * compatible with each other */

    ASSERT_TRUE( !arch_info_a.isArchIdenticalTo( arch_info_b ) );
    ASSERT_TRUE( !arch_info_a.isArchIdenticalTo( arch_info_b.archId() ) );

    ASSERT_TRUE( !arch_info_a.isArchCompatibleWith(arch_info_b ) );
    ASSERT_TRUE( !arch_info_a.isArchCompatibleWith(arch_info_b.archId() ) );
}

/* end: tests/sixtracklib/common/control/test_arch_info_c99.cpp */
