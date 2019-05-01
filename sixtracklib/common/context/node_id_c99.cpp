#include "sixtracklib/common/context/node_id.h"
#include "sixtracklib/common/context/node_id.hpp"

NS(node_platform_id_t) NS(NodeId_get_platform_id)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( node_id != nullptr )
        ? node_id->platformId()
        : SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_PATFORM_ID;
}

NS(node_device_id_t) NS(NodeId_get_device_id)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( node_id != nullptr )
        ? node_id->deviceId()
        : SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_DEVICE_ID;
}

bool NS(NodeId_has_node_index)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( ( node_id != nullptr ) && ( node_id->hasIndex() ) );
}

NS(node_index_t) NS(NodeId_get_node_index)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( node_id != nullptr )
        ? node_id->index() : SIXTRL_CXX_NAMESPACE::NODE_UNDEFINED_INDEX;
}

void NS(NodeId_clear)(
    SIXTRL_ARGPTR_DEC NS(NodeId)* SIXTRL_RESTRICT node_id )
{
    if( node_id != nullptr ) node_id->clear();
}

void NS(NodeId_reset)(
    SIXTRL_ARGPTR_DEC NS(NodeId)* SIXTRL_RESTRICT node_id,
    NS(node_platform_id_t) const platform_id,
    NS(node_device_id_t) const device_id,
    NS(node_index_t) const node_index )
{
    if( node_id != nullptr )
    {
        node_id->reset( platform_id, device_id, node_index );
    }
}

void NS(NodeId_set_platform_id)(
    SIXTRL_ARGPTR_DEC NS(NodeId)* SIXTRL_RESTRICT node_id,
    NS(node_platform_id_t) const platform_id )
{
    if( node_id != nullptr ) node_id->setPlatformId( platform_id );
}

void NS(NodeId_set_device_id)(
    SIXTRL_ARGPTR_DEC NS(NodeId)* SIXTRL_RESTRICT node_id,
    NS(node_device_id_t) const device_id )
{
    if( node_id != nullptr ) node_id->setDeviceId( device_id );
}

void NS(NodeId_set_index)(
    SIXTRL_ARGPTR_DEC NS(NodeId)* SIXTRL_RESTRICT node_id,
    NS(node_index_t) const node_index )
{
    if( node_id != nullptr ) node_id->setNodeIndex( node_index );
}

bool NS(NodeId_to_string)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT node_id,
    SIXTRL_ARGPTR_DEC char* SIXTRL_RESTRICT node_id_str,
    NS(buffer_size_t) const node_id_str_capacity )
{
    return ( ( node_id != nullptr ) &&
             ( node_id->toString( node_id_str, node_id_str_capacity ) ) );
}

bool NS(NodeId_from_string)(
    SIXTRL_ARGPTR_DEC NS(NodeId)* SIXTRL_RESTRICT node_id,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT node_id_str )
{
    return ( ( node_id != nullptr ) &&
             ( node_id->fromString( node_id_str ) ) );
}

int NS(NodeId_compare)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( ( lhs != nullptr ) && ( rhs != nullptr ) )
    {
        cmp_result = SIXTRL_CXX_NAMESPACE::compareNodeIds( *lhs, *rhs );
    }
    else if( rhs != nullptr )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

bool NS(NodeId_are_equal)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT rhs )
{
    return ( NS(NodeId_compare)( lhs, rhs ) == 0 );
}

void NS(NodeId_print_out)(
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    NS(NodeId_print)( stdout, node_id );
}

void NS(NodeId_print)( FILE* SIXTRL_RESTRICT output,
    SIXTRL_ARGPTR_DEC const NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    if( node_id != nullptr )
    {
        SIXTRL_CXX_NAMESPACE::printNodeId( output, *node_id );
    }
}

/* end: sixtracklib/common/context/node_id_c99.cpp */
