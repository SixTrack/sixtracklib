#include "node_info.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/node_info.hpp"


#if !defined( _GPUCODE )

namespace st = SIXTRL_CXX_NAMESPACE;

void NS(NodeInfo_delete)( ::NS(NodeInfoBase)* SIXTRL_RESTRICT node_info )
{
    if( node_info != nullptr ) delete node_info;
}

/* ------------------------------------------------------------------------- */

::NS(NodeId) const* NS(NodeInfo_get_ptr_const_node_id)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrNodeId() : nullptr;
}

::NS(NodeId)* NS(NodeInfo_get_ptr_node_id)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrNodeId() : nullptr;
}

::NS(node_platform_id_t) NS(NodeInfo_get_platform_id)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr )
        ? info->platformId() : st::NODE_ILLEGAL_PATFORM_ID;
}

void NS(NodeInfo_set_platform_id)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    ::NS(node_platform_id_t) const platform_id )
{
    if( info != nullptr ) info->setPlatformId( platform_id );
}

::NS(node_device_id_t) NS(NodeInfo_get_device_id)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr )
        ? info->deviceId() : st::NODE_ILLEGAL_DEVICE_ID;
}

void NS(NodeInfo_set_device_id)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    ::NS(node_device_id_t) const device_id )
{
    if( info != nullptr ) info->setDeviceId( device_id );
}

bool NS(NodeInfo_has_node_index)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasNodeIndex() ) );
}

::NS(node_index_t) NS(NodeInfo_get_node_index)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr )
        ? info->nodeIndex() : st::NODE_UNDEFINED_INDEX;
}

void NS(NodeInfo_set_node_index)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    ::NS(node_index_t) node_index )
{
    if( info != nullptr ) info->setNodeIndex( node_index );
}

bool NS(NodeInfo_is_default_node)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->isDefaultNode() ) );
}

bool NS(NodeInfo_is_selected_node)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->isSelectedNode() ) );
}

void NS(NodeInfo_set_is_default_node)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    bool const is_default )
{
    if( info != nullptr ) info->setIsDefaultNode( is_default );
}

void NS(NodeInfo_set_is_selected_node)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    bool const is_selected )
{
    if( info != nullptr ) info->setIsSelectedNode( is_selected );
}

/* ------------------------------------------------------------------------- */

::NS(arch_id_t) NS(NodeInfo_get_arch_id)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->archId() : ::NS(ARCHITECTURE_ILLEGAL);
}

bool NS(NodeInfo_has_arch_string)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasArchStr() ) );
}

char const* NS(NodeInfo_get_arch_string)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrArchStr() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(NodeInfo_has_platform_name)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasPlatformName() ) );
}

char const* NS(NodeInfo_get_platform_name)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrPlatformNameStr() : nullptr;
}

void NS(NodeInfo_set_platform_name)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT platform_name )
{
    if( info != nullptr ) info->setPlatformName( platform_name );
}

bool NS(NodeInfo_has_device_name)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasDeviceName() ) );
}

char const* NS(NodeInfo_get_device_name)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrDeviceNameStr() : nullptr;
}

void NS(NodeInfo_set_device_name)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT device_name )
{
    if( info != nullptr ) info->setDeviceName( device_name );
}

bool NS(NodeInfo_has_description)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasDescription() ) );
}

char const* NS(NodeInfo_get_description)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrDescriptionStr() : nullptr;
}

void NS(NodeInfo_set_description)(
    SIXTRL_ARGPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT description )
{
    if( info != nullptr ) info->setPlatformName( description );
}

/* ------------------------------------------------------------------------- */

void NS(NodeInfo_print)( SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const
    SIXTRL_RESTRICT info, ::FILE* SIXTRL_RESTRICT output )
{
    if( ( info != nullptr ) && ( output != nullptr ) )
    {
        info->print( output );
    }
}

void NS(NodeInfo_print_out)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    if( info != nullptr ) info->printOut();
}


::NS(arch_size_t) NS(NodeInfo_get_required_output_str_length)(
    SIXTRL_ARGPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) 
        ? info->requiredOutStringLength() : ::NS(arch_size_t){ 0 };    
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(NodeInfo_convert_to_string)(
    SIXTRL_ARGPTR_DEC const NS(NodeInfoBase) *const SIXTRL_RESTRICT info, 
    NS(arch_size_t) const out_string_capacity,
    char* SIXTRL_RESTRICT out_string )
{
    return ( info != nullptr ) 
        ? info->toString( out_string_capacity, out_string ) 
        : st::ARCH_STATUS_GENERAL_FAILURE;    
}

#endif /* !defined( _GPUCODE ) */

/* end: sixtracklib/common/control/node_info_c99.cpp */
