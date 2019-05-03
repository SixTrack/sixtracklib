#include "sixtracklib/common/control/node_id.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/testlib.h"

TEST( CXX_CommonControlNodeIdTests, MinimalUsage )
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    using node_id_t     = st::NodeId;
    using ctrl_size_t   = st::controller_size_t;
    using platform_id_t = node_id_t::platform_id_t;
    using device_id_t   = node_id_t::device_id_t;
    using node_index_t  = node_id_t::index_t;

    char node_id_str[] = { '\0', '\0', '\0', '\0' };

    static constexpr ctrl_size_t TOO_SHORT_ID_STR_CAPACITY = ctrl_size_t{ 3 };
    static constexpr ctrl_size_t NODE_ID_STR_CAPACITY = ctrl_size_t{ 4 };

    /* Create an empty, non-initialized nodeId instance */

    node_id_t node_id_a;

    /* Verify that all properties are set to the initial state */

    ASSERT_TRUE( !node_id_a.valid() );
    ASSERT_TRUE(  node_id_a.platformId() == st::NODE_ILLEGAL_PATFORM_ID );
    ASSERT_TRUE(  node_id_a.deviceId()   == st::NODE_ILLEGAL_DEVICE_ID );
    ASSERT_TRUE( !node_id_a.hasIndex() );
    ASSERT_TRUE(  node_id_a.index()  == st::NODE_UNDEFINED_INDEX );

    /* creating a node_id string from an illegal nodeId should never work,
       regardless whether the buffer is large enough or not */

    ASSERT_TRUE( !node_id_a.toString(
        node_id_str, TOO_SHORT_ID_STR_CAPACITY ) );

    ASSERT_TRUE( !node_id_a.toString( node_id_str, NODE_ID_STR_CAPACITY ) );

    /* Setting the platform_id to a valid value should not change the
       overall picture much: */

    node_id_a.setPlatformId( platform_id_t{ 0 } );

    ASSERT_TRUE( !node_id_a.valid() );
    ASSERT_TRUE( node_id_a.platformId() == platform_id_t{ 0 } );

    ASSERT_TRUE( !node_id_a.toString(
        node_id_str, TOO_SHORT_ID_STR_CAPACITY ) );

    ASSERT_TRUE( !node_id_a.toString( node_id_str, NODE_ID_STR_CAPACITY ) );

    /* Setting also the device makes however a difference: */

    node_id_a.setDeviceId( device_id_t{ 0 } );

    ASSERT_TRUE( node_id_a.valid() );
    ASSERT_TRUE( node_id_a.platformId() == platform_id_t{ 0 } );
    ASSERT_TRUE( node_id_a.deviceId() == device_id_t{ 0 } );

    /* The node index has never been touched, verify this */

    ASSERT_TRUE( !node_id_a.hasIndex() );
    ASSERT_TRUE(  node_id_a.index() == st::NODE_UNDEFINED_INDEX );

    /* Now we should be able to create a node_id_str if we provide a
       large enough string buffer */

    ASSERT_TRUE( !node_id_a.toString(
        node_id_str, TOO_SHORT_ID_STR_CAPACITY ) );

    ASSERT_TRUE( node_id_a.toString( node_id_str, NODE_ID_STR_CAPACITY ) );
    ASSERT_TRUE( std::strcmp( node_id_str, "0.0" ) == 0 );

    std::string const cxx_node_id_str = node_id_a.toString();
    ASSERT_TRUE( cxx_node_id_str.compare( node_id_str ) == 0 );

    /* Create a second nodeId instance from the node_id_str */

    node_id_t node_id_b( cxx_node_id_str );

    /* Verify that both node_id_a and node_id_b are identical */

    ASSERT_TRUE( 0 == st::compareNodeIds( node_id_a, node_id_b ) );
    ASSERT_TRUE( node_id_a.platformId() == node_id_b.platformId() );
    ASSERT_TRUE( node_id_a.deviceId() == node_id_b.deviceId() );

    /* Since the node index is not part of the node_id_str, there should
     * be no index set in node_id_b  */

    ASSERT_TRUE( !node_id_b.hasIndex() );

    /* Setting an index in node_id_a should not change the identiy relations
     * for similar reasons */

    node_id_a.setIndex( node_index_t{ 42 } );

    ASSERT_TRUE( node_id_a.hasIndex() );
    ASSERT_TRUE( node_id_a.index() == node_index_t{ 42 } );

    ASSERT_TRUE( 0 == st::compareNodeIds( node_id_a, node_id_b ) );
    ASSERT_TRUE( node_id_a.platformId() == node_id_b.platformId() );
    ASSERT_TRUE( node_id_a.deviceId() == node_id_b.deviceId() );

    ASSERT_TRUE( !node_id_b.hasIndex() );
    ASSERT_TRUE( node_id_b.index() != node_id_a.index() );

    /* Changing the platform and/or the device id on one of the nodes will
     * change equality relations */

    node_id_b.setPlatformId( platform_id_t{ 1 } );

    ASSERT_TRUE(  0 != st::compareNodeIds( node_id_a, node_id_b ) );
    ASSERT_TRUE( -1 == st::compareNodeIds( node_id_a, node_id_b ) );

    ASSERT_TRUE( node_id_a.platformId() != node_id_b.platformId() );
    ASSERT_TRUE( node_id_a.deviceId() == node_id_b.deviceId() );

    /* Update node_id_a to the same state as node_id_b via a node_id_str */

    ASSERT_TRUE( node_id_b.toString( node_id_str, NODE_ID_STR_CAPACITY ) );
    ASSERT_TRUE( 0 == std::strcmp( node_id_str, "1.0" ) );
    ASSERT_TRUE( node_id_a.fromString( node_id_str ) );

    /* Verify that node_id_a is now again equivalent to node_id_b and that
     * the operation has not changed the node index of node_id_a */

    ASSERT_TRUE( 0 == st::compareNodeIds( node_id_a, node_id_b ) );
    ASSERT_TRUE( node_id_a.platformId() == node_id_b.platformId() );
    ASSERT_TRUE( node_id_a.deviceId() == node_id_b.deviceId() );

    ASSERT_TRUE( node_id_a.hasIndex() );
    ASSERT_TRUE( node_id_a.index() == node_index_t{ 42 } );
}

/* end: tests/sixtracklib/common/control/test_node_id_c99.cpp */
