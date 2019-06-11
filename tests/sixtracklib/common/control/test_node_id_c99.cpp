#include "sixtracklib/common/control/node_id.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/testlib.h"

TEST( C99_CommonControlNodeIdTests, MinimalUsage )
{
    using node_id_t     = ::NS(NodeId);
    using platform_id_t = ::NS(node_platform_id_t);
    using device_id_t   = ::NS(node_device_id_t);
    using node_index_t  = ::NS(node_index_t);
    using ctrl_size_t   = ::NS(ctrl_size_t);

    char node_id_str[] = { '\0', '\0', '\0', '\0' };

    static constexpr ctrl_size_t TOO_SHORT_ID_STR_CAPACITY = ctrl_size_t{ 3 };
    static constexpr ctrl_size_t NODE_ID_STR_CAPACITY = ctrl_size_t{ 4 };

    /* Create an empty, non-initialized nodeId instance */

    node_id_t* node_id_a = ::NS(NodeId_create)();
    ASSERT_TRUE(  node_id_a != nullptr );

    /* Verify that all properties are set to the initial state */

    ASSERT_TRUE( !::NS(NodeId_is_valid)( node_id_a ) );

    ASSERT_TRUE(  ::NS(NodeId_get_platform_id)( node_id_a ) ==
                  ::NS(NODE_ILLEGAL_PATFORM_ID) );

    ASSERT_TRUE(  ::NS(NodeId_get_device_id)( node_id_a ) ==
                  ::NS(NODE_ILLEGAL_DEVICE_ID) );

    ASSERT_TRUE( !::NS(NodeId_has_node_index)( node_id_a ) );
    ASSERT_TRUE(  ::NS(NodeId_get_node_index)( node_id_a ) ==
                  ::NS(NODE_UNDEFINED_INDEX) );

    /* creating a node_id string from an illegal nodeId should never work,
       regardless whether the buffer is large enough or not */

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(NodeId_to_string)(
        node_id_a, node_id_str, TOO_SHORT_ID_STR_CAPACITY ) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(NodeId_to_string)(
        node_id_a, node_id_str, NODE_ID_STR_CAPACITY ) );

    /* Setting the platform_id to a valid value should not change the
       overall picture much: */

    ::NS(NodeId_set_platform_id)( node_id_a, platform_id_t{ 0 } );

    ASSERT_TRUE( !::NS(NodeId_is_valid)( node_id_a ) );
    ASSERT_TRUE(  ::NS(NodeId_get_platform_id)( node_id_a ) ==
                  platform_id_t{ 0 } );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(NodeId_to_string)(
        node_id_a, node_id_str, TOO_SHORT_ID_STR_CAPACITY ) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(NodeId_to_string)(
        node_id_a, node_id_str, NODE_ID_STR_CAPACITY ) );

    /* Setting also the device makes however a difference: */

    ::NS(NodeId_set_device_id)( node_id_a, device_id_t{ 0 } );

    ASSERT_TRUE( ::NS(NodeId_is_valid)( node_id_a ) );

    ASSERT_TRUE( ::NS(NodeId_get_platform_id)( node_id_a ) ==
                 platform_id_t{ 0 } );

    ASSERT_TRUE( ::NS(NodeId_get_device_id)( node_id_a ) ==
                 device_id_t{ 0 } );

    /* The node index has never been touched, verify this */

    ASSERT_TRUE( !::NS(NodeId_has_node_index)( node_id_a ) );
    ASSERT_TRUE(  ::NS(NodeId_get_node_index)( node_id_a ) ==
                  ::NS(NODE_UNDEFINED_INDEX) );

    /* Now we should be able to create a node_id_str if we provide a
       large enough string buffer */

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) != ::NS(NodeId_to_string)(
        node_id_a, node_id_str, TOO_SHORT_ID_STR_CAPACITY ) );

    ASSERT_TRUE(  ::NS(ARCH_STATUS_SUCCESS) == ::NS(NodeId_to_string)(
        node_id_a, node_id_str, NODE_ID_STR_CAPACITY ) );

    ASSERT_TRUE(  std::strcmp( node_id_str, "0.0" ) == 0 );

    /* Create a second nodeId instance from the node_id_str */

    node_id_t* node_id_b = ::NS(NodeId_new_from_string)( node_id_str );
    ASSERT_TRUE( node_id_b != nullptr );

    /* Verify that both node_id_a and node_id_b are identical */

    ASSERT_TRUE( 0 == ::NS(NodeId_compare)( node_id_a, node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_get_platform_id)( node_id_a ) ==
                 ::NS(NodeId_get_platform_id)( node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_get_device_id)( node_id_a ) ==
                 ::NS(NodeId_get_device_id)( node_id_b ) );

    /* Since the node index is not part of the node_id_str, there should
     * be no index set in node_id_b  */

    ASSERT_TRUE( !::NS(NodeId_has_node_index)( node_id_b ) );

    /* Setting an index in node_id_a should not change the identiy relations
     * for similar reasons */

    ::NS(NodeId_set_index)( node_id_a, node_index_t{ 42 } );

    ASSERT_TRUE( ::NS(NodeId_has_node_index)( node_id_a ) );
    ASSERT_TRUE( ::NS(NodeId_get_node_index)( node_id_a ) ==
                 node_index_t{ 42 } );

    ASSERT_TRUE( 0 == ::NS(NodeId_compare)( node_id_a, node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_get_platform_id)( node_id_a ) ==
                 ::NS(NodeId_get_platform_id)( node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_get_device_id)( node_id_a ) ==
                 ::NS(NodeId_get_device_id)( node_id_b ) );

    ASSERT_TRUE( !::NS(NodeId_has_node_index)( node_id_b ) );
    ASSERT_TRUE(  ::NS(NodeId_get_node_index)( node_id_b ) !=
                  ::NS(NodeId_get_node_index)( node_id_a ) );

    /* Changing the platform and/or the device id on one of the nodes will
     * change equality relations */

    ::NS(NodeId_set_platform_id)( node_id_b, platform_id_t{ 1 } );

    ASSERT_TRUE(  0 != ::NS(NodeId_compare)( node_id_a, node_id_b ) );
    ASSERT_TRUE( -1 == ::NS(NodeId_compare)( node_id_a, node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_get_platform_id)( node_id_a ) !=
                 ::NS(NodeId_get_platform_id)( node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_get_device_id)( node_id_a ) ==
                 ::NS(NodeId_get_device_id)( node_id_b ) );

    /* Update node_id_a to the same state as node_id_b via a node_id_str */

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(NodeId_to_string)(
        node_id_b, node_id_str, NODE_ID_STR_CAPACITY ) );

    ASSERT_TRUE( 0 == std::strcmp( node_id_str, "1.0" ) );

    ASSERT_TRUE( ::NS(ARCH_STATUS_SUCCESS) == ::NS(NodeId_from_string)(
        node_id_a, node_id_str ) );

    /* Verify that node_id_a is now again equivalent to node_id_b and that
     * the operation has not changed the node index of node_id_a */

    ASSERT_TRUE( 0 == ::NS(NodeId_compare)( node_id_a, node_id_b ) );
    ASSERT_TRUE( ::NS(NodeId_get_platform_id)( node_id_a ) ==
                 ::NS(NodeId_get_platform_id)( node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_get_device_id)( node_id_a ) ==
                 ::NS(NodeId_get_device_id)( node_id_b ) );

    ASSERT_TRUE( ::NS(NodeId_has_node_index)( node_id_a ) );
    ASSERT_TRUE( ::NS(NodeId_get_node_index)( node_id_a ) ==
                 node_index_t{ 42 } );

    /* Cleanup */

    ::NS(NodeId_delete)( node_id_a );
    node_id_a = nullptr;

    ::NS(NodeId_delete)( node_id_b );
    node_id_b = nullptr;
}


TEST( C99_CommonControlNodeIdTests, ExtractNodeIdFromConfigStr )
{
    using buf_size_t = ::NS(buffer_size_t);
    using status_t   = ::NS(ctrl_status_t);
    std::string conf_str( "" );

    ::NS(buffer_size_t) const max_out_str_len = 32u;
    char device_id_str[ 32 ];
    std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );

    status_t ret = ::NS(NodeId_extract_node_id_str_from_config_str)(
        conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );

    ASSERT_TRUE( ret == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( std::strlen( &device_id_str[ 0 ] ) == buf_size_t{ 0 } );

    conf_str = "0.0";
    std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );

    ret = ::NS(NodeId_extract_node_id_str_from_config_str)(
        conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );

    ASSERT_TRUE( ret == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( std::strcmp( &device_id_str[ 0 ], "0.0" ) == 0 );

    conf_str = "  0.0  ";
    std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );

    ret = ::NS(NodeId_extract_node_id_str_from_config_str)(
        conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );

    ASSERT_TRUE( ret == ::NS(ARCH_STATUS_SUCCESS) );
    ASSERT_TRUE( std::strcmp( &device_id_str[ 0 ], "0.0" ) == 0 );

//     conf_str = "0.0;a=b;#this is a comment";
//     std::memset( &device_id_str[ 0 ], ( int )'\0', max_out_str_len );
//
//     ret = ::NS(NodeId_extract_node_id_str_from_config_str)(
//         conf_str.c_str(), &device_id_str[ 0 ], max_out_str_len );
//
//     ASSERT_TRUE( ret == 0 );
//     ASSERT_TRUE( std::strcmp( &device_id_str[ 0 ], "0.0" ) == 0 );
}

/* end: tests/sixtracklib/common/control/test_node_id_c99.cpp */
