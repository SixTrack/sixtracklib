#include "sixtracklib/common/be_dipedge/be_dipedge.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.h"

TEST( C99CommonBeamElementDipoleEdgeTests, BasicUsage )
{
    using be_dipedge_t = ::NS(DipoleEdge);
    using buffer_t     = ::NS(Buffer);
    using size_t       = ::NS(buffer_size_t);

    be_dipedge_t e1;
    ::NS(DipoleEdge_preset)( &e1 );

    buffer_t* eb = ::NS(Buffer_new)( size_t{ 0 } );
    SIXTRL_ASSERT( eb != nullptr );

    be_dipedge_t* e2 = ::NS(DipoleEdge_new)( eb );
    ASSERT_TRUE( e2 != nullptr );

    be_dipedge_t* e3 = ::NS(DipoleEdge_add)( eb, 1.0, 2.0 );
    ASSERT_TRUE( e3 != nullptr );

    be_dipedge_t* e4 = ::NS(DipoleEdge_add_copy)( eb, &e1 );
    ASSERT_TRUE( e4 != nullptr );
    ASSERT_TRUE( 0 == ::NS(DipoleEdge_compare_values)( e4, &e1 ) );

    ::NS(Buffer_delete)( eb );
}

/* end: tests/sixtracklib/common/beam_elements/test_be_limit_c99.cpp */
