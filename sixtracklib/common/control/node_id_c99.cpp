#include "sixtracklib/common/control/node_id.h"
#include "sixtracklib/common/control/node_id.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

SIXTRL_ARGPTR_DEC ::NS(NodeId)*
NS(NodeId_preset)( SIXTRL_ARGPTR_DEC ::NS(NodeId)* SIXTRL_RESTRICT node_id )
{
    if( node_id != nullptr )
    {
        node_id->clear();
    }

    return node_id;
}

SIXTRL_ARGPTR_DEC ::NS(NodeId)* NS(NodeId_create)( void )
{
    return new st::NodeId;
}

void NS(NodeId_delete)( SIXTRL_ARGPTR_DEC NS(NodeId)* SIXTRL_RESTRICT node_id )
{
    delete node_id;
    return;
}

SIXTRL_ARGPTR_DEC ::NS(NodeId)* NS(NodeId_new)(
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id )
{
    return new st::NodeId( platform_id, device_id );
}

SIXTRL_ARGPTR_DEC ::NS(NodeId)* NS(NodeId_new_from_string)(
    char const* SIXTRL_RESTRICT node_id_str )
{
    return ( ( node_id_str != nullptr ) &&
             ( std::strlen( node_id_str ) > std::size_t{ 0 } ) )
        ? new st::NodeId( node_id_str ) : nullptr;
}

SIXTRL_ARGPTR_DEC ::NS(NodeId)* NS(NodeId_new_detailed)(
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id,
    ::NS(node_index_t) const node_index )
{
    return new st::NodeId( platform_id, device_id, node_index );
}

bool NS(NodeId_is_valid)( SIXTRL_ARGPTR_DEC
    const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( ( node_id != nullptr ) && ( node_id->valid() ) );
}

::NS(node_platform_id_t) NS(NodeId_get_platform_id)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( node_id != nullptr )
        ? node_id->platformId() : ::NS(NODE_ILLEGAL_PATFORM_ID);
}

::NS(node_device_id_t) NS(NodeId_get_device_id)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( node_id != nullptr )
        ? node_id->deviceId() : ::NS(NODE_ILLEGAL_DEVICE_ID);
}

bool NS(NodeId_has_node_index)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( ( node_id != nullptr ) && ( node_id->hasIndex() ) );
}

::NS(node_index_t) NS(NodeId_get_node_index)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    return ( node_id != nullptr )
        ? node_id->index() : ::NS(NODE_UNDEFINED_INDEX);
}

void NS(NodeId_clear)(
    SIXTRL_ARGPTR_DEC ::NS(NodeId)* SIXTRL_RESTRICT node_id )
{
    if( node_id != nullptr ) node_id->clear();
}

void NS(NodeId_reset)(
    SIXTRL_ARGPTR_DEC ::NS(NodeId)* SIXTRL_RESTRICT node_id,
    ::NS(node_platform_id_t) const platform_id,
    ::NS(node_device_id_t) const device_id,
    ::NS(node_index_t) const node_index )
{
    if( node_id != nullptr )
    {
        node_id->reset( platform_id, device_id, node_index );
    }
}

void NS(NodeId_set_platform_id)(
    SIXTRL_ARGPTR_DEC ::NS(NodeId)* SIXTRL_RESTRICT node_id,
    ::NS(node_platform_id_t) const platform_id )
{
    if( node_id != nullptr ) node_id->setPlatformId( platform_id );
}

void NS(NodeId_set_device_id)(
    SIXTRL_ARGPTR_DEC ::NS(NodeId)* SIXTRL_RESTRICT node_id,
    ::NS(node_device_id_t) const device_id )
{
    if( node_id != nullptr ) node_id->setDeviceId( device_id );
}

void NS(NodeId_set_index)(
    SIXTRL_ARGPTR_DEC ::NS(NodeId)* SIXTRL_RESTRICT node_id,
    ::NS(node_index_t) const node_index )
{
    if( node_id != nullptr ) node_id->setIndex( node_index );
}

bool NS(NodeId_to_string)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT node_id,
    SIXTRL_ARGPTR_DEC char* SIXTRL_RESTRICT node_id_str,
    ::NS(buffer_size_t) const node_id_str_capacity )
{
    return ( ( node_id != nullptr ) &&
             ( node_id->toString( node_id_str, node_id_str_capacity ) ) );
}

bool NS(NodeId_from_string)(
    SIXTRL_ARGPTR_DEC ::NS(NodeId)* SIXTRL_RESTRICT node_id,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT node_id_str )
{
    return ( ( node_id != nullptr ) &&
             ( node_id->fromString( node_id_str ) ) );
}

int NS(NodeId_compare)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT rhs )
{
    int cmp_result = -1;

    if( ( lhs != nullptr ) && ( rhs != nullptr ) )
    {
        cmp_result = st::compareNodeIds( *lhs, *rhs );
    }
    else if( rhs != nullptr )
    {
        cmp_result = +1;
    }

    return cmp_result;
}

bool NS(NodeId_are_equal)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT lhs,
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT rhs )
{
    return ( ::NS(NodeId_compare)( lhs, rhs ) == 0 );
}

/* ------------------------------------------------------------------------- */

void NS(NodeId_print_out)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    ::NS(NodeId_print)( stdout, node_id );
}

void NS(NodeId_print)( ::FILE* SIXTRL_RESTRICT output,
    SIXTRL_ARGPTR_DEC const ::NS(NodeId) *const SIXTRL_RESTRICT node_id )
{
    if( node_id != nullptr )
    {
        st::printNodeId( output, *node_id );
    }
}

::NS(ctrl_status_t) NS(NodeId_extract_node_id_str_from_config_str)(
    char const* SIXTRL_RESTRICT config_str, char* SIXTRL_RESTRICT node_id_str,
    ::NS(buffer_size_t) const node_id_str_capacity )
{
    int success = ::NS(ARCH_STATUS_GENERAL_FAILURE);
    using buf_size_t = ::NS(buffer_size_t);

    if( ( config_str != nullptr ) && ( node_id_str != nullptr ) &&
        ( std::strlen( config_str ) > buf_size_t{ 0 } ) &&
        ( node_id_str_capacity > buf_size_t{ 1 } ) )
    {
        std::memset( node_id_str, ( int )'\0', node_id_str_capacity );
        std::string const str = st::NodeId_extract_node_id_str_from_config_str(
            config_str );

        if( !str.empty() )
        {
            if( str.size() < node_id_str_capacity )
            {
                std::strncpy( node_id_str, str.c_str(), str.size() );
                success = ::NS(ARCH_STATUS_SUCCESS);
            }
        }
        else
        {
            success = ::NS(ARCH_STATUS_SUCCESS);
        }
    }
    else if( ( config_str != nullptr ) &&
             ( std::strlen( config_str ) == buf_size_t{ 0 } ) )
    {
        if( ( node_id_str != nullptr ) &&
            ( node_id_str_capacity > buf_size_t{ 1 } ) )
        {
            std::memset( node_id_str, ( int )'\0', node_id_str_capacity );
        }

        success = ::NS(ARCH_STATUS_SUCCESS);
    }

    return success;
}

/* end: sixtracklib/common/control/node_id_c99.cpp */
