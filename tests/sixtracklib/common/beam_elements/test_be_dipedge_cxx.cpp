#include "sixtracklib/common/be_dipedge/be_dipedge.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "sixtracklib/testlib.h"

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/buffer.hpp"

TEST( CXXCommonBeamElementDipoleEdgeTests, BasicUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using be_dipedge_t = st::DipoleEdge;
    using buffer_t     = st::Buffer;

    be_dipedge_t e1;
    e1.preset();

    buffer_t eb;

    be_dipedge_t* e2 = eb.createNew< be_dipedge_t >();
    ASSERT_TRUE( e2 != nullptr );

    be_dipedge_t* e3 = eb.add< be_dipedge_t >( 1.0, 2.0 );
    ASSERT_TRUE( e3 != nullptr );

    be_dipedge_t* e4 = eb.addCopy( e1 );
    ASSERT_TRUE( e4 != nullptr );

    ASSERT_TRUE( 0 == ::NS(DipoleEdge_compare_values)(
        e4->getCApiPtr(), e1.getCApiPtr() ) );
}

/* end: tests/sixtracklib/common/beam_elements/test_be_limit_cxx.cpp */
