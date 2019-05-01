#include "node_info.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/node_info.hpp"


#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */

::NS(NodeId) const* NS(NodeInfo_get_ptr_node_id)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrNodeId() : nullptr;
}

::NS(node_platform_id_t) NS(NodeInfo_get_platform_id)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr )
        ? info->platformId() : SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_PATFORM_ID;
}

::NS(node_device_id_t) NS(NodeInfo_get_device_id)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr )
        ? info->deviceId() : SIXTRL_CXX_NAMESPACE::NODE_ILLEGAL_DEVICE_ID;
}

bool NS(NodeInfo_is_default_node)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->isDefaultNode() ) );
}

/* ------------------------------------------------------------------------- */

::NS(arch_id_t) NS(NodeInfo_get_arch_id)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr )
        ? info->archId() : SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_INVALID;
}

bool NS(NodeInfo_has_arch_string)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasArchStr() ) );
}

char const* NS(NodeInfo_get_arch_string)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrArchStr() : nullptr;
}

/* ------------------------------------------------------------------------- */

bool NS(NodeInfo_has_platform_name)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasPlatformName() ) );
}

char const* NS(NodeInfo_get_platform_name)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrPlatformNameStr() : nullptr;
}

bool NS(NodeInfo_has_device_name)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasDeviceName() ) );
}

char const* NS(NodeInfo_get_device_name)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    retrn ( info != nullptr ) ? info->ptrDeviceNameStr() : nullptr;
}

bool NS(NodeInfo_has_description)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( ( info != nullptr ) && ( info->hasDescription() ) );
}

char const* NS(NodeInfo_get_description)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    return ( info != nullptr ) ? info->ptrDescriptionStr() : nullptr;
}

/* ------------------------------------------------------------------------- */

void NS(NodeInfo_reset)(
    SIXTR_ARGPPTR_DEC ::NS(NodeInfoBase)* SIXTRL_RESTRICT info,
    ::NS(arch_id_t) const arch_id, const char *const SIXTRL_RESTRICT arch_str,
    ::NS(platform_id_t) const platform_id, ::NS(device_id_t) const device_id,
    const char *const SIXTRL_RESTRICT platform_name,
    const char *const SIXTRL_RESTRICT device_name,
    const char *const SIXTRL_RESTRICT description )
{
    if( info != nullptr )
    {
        info->markAsDefault( false );
        info->reset( arch_id, arch_str, platform_id, device_id, platform_name,
                     device_name, description );
    }
}

/* ------------------------------------------------------------------------- */

void NS(NodeInfo_print)( SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const
    SIXTRL_RESTRICT info, ::FILE* SIXTRL_RESTRICT output )
{
    if( ( info != nullptr ) && ( output != nullptr ) )
    {
        info->print( output );
    }
}

void NS(NodeInfo_print_out)(
    SIXTR_ARGPPTR_DEC const ::NS(NodeInfoBase) *const SIXTRL_RESTRICT info )
{
    if( info != nullptr ) info->printOut();
}

/* ------------------------------------------------------------------------- */

void NS(NodeInfo_mark_as_default)( SIXTR_ARGPPTR_DEC ::NS(NodeInfoBase)*
    SIXTRL_RESTRICT info, bool const is_default )
                                 )
{
    if( info != nullptr ) info->markAsDefault( is_default );
}

#endif /* !defined( _GPUCODE ) */

/* end: sixtracklib/common/context/node_info_c99.cpp */
