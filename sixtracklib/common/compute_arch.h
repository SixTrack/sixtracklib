#ifndef SIXTRACKLIB_COMMON_COMPUTE_ARCH_H__
#define SIXTRACKLIB_COMMON_COMPUTE_ARCH_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef SIXTRL_INT64_T NS(comp_node_id_num_t);

typedef struct NS(ComputeNodeId)
{
    NS(comp_node_id_num_t)  platform_id;
    NS(comp_node_id_num_t)  device_id;
}
NS(ComputeNodeId);

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if defined( __cplusplus )

bool operator<( NS(ComputeNodeId) const& lhs,
                NS(ComputeNodeId) const& rhs ) noexcept
{
    return ( ( lhs.platform_id < rhs.platform_id ) ||
             ( ( lhs.platform_id == rhs.platform_id ) &&
               ( lhs.device_id   <  rhs.device_id   ) ) );
}

#endif /* !defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_HOST_FN NS(ComputeNodeId)* NS(ComputeNodeId_preset)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id );

SIXTRL_HOST_FN SIXTRL_STATIC NS(comp_node_id_num_t)
NS(ComputeNodeId_get_platform_id)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id );

SIXTRL_HOST_FN SIXTRL_STATIC NS(comp_node_id_num_t)
NS(ComputeNodeId_get_device_id)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id );

SIXTRL_HOST_FN SIXTRL_STATIC void NS(ComputeNodeId_set_platform_id)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    NS(comp_node_id_num_t) const platform_id );

SIXTRL_HOST_FN SIXTRL_STATIC void NS(ComputeNodeId_set_device_id)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    NS(comp_node_id_num_t) const device_id );

SIXTRL_HOST_FN SIXTRL_STATIC int NS(ComputeNodeId_is_valid)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id );


typedef struct NS(ComputeNodeInfo)
{
    NS(ComputeNodeId) id;

    char*   arch;
    char*   platform;
    char*   name;
    char*   description;
}
NS(ComputeNodeInfo);

SIXTRL_HOST_FN NS(ComputeNodeInfo)* NS(ComputeNodeInfo_preset)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN void NS(ComputeNodeInfo_free)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN NS(ComputeNodeInfo)* NS(ComputeNodeInfo_reserve)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    SIXTRL_SIZE_T const arch_str_len,
    SIXTRL_SIZE_T const platform_str_len,
    SIXTRL_SIZE_T const name_str_len,
    SIXTRL_SIZE_T const description_str_len );

SIXTRL_HOST_FN int NS(ComputeNodeInfo_make)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    NS(ComputeNodeId) const id, const char *const SIXTRL_RESTRICT arch_str,
    const char *const SIXTRL_RESTRICT platform_str,
    const char *const SIXTRL_RESTRICT name_str,
    const char *const SIXTRL_RESTRICT description_str );

SIXTRL_HOST_FN SIXTRL_STATIC int NS(ComputeNodeInfo_is_valid)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN SIXTRL_STATIC NS(ComputeNodeId)
NS(ComputeNodeInfo_get_id)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN SIXTRL_STATIC NS(comp_node_id_num_t)
NS(ComputeNodeInfo_get_platform_id)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN SIXTRL_STATIC NS(comp_node_id_num_t)
NS(ComputeNodeInfo_get_device_id)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN SIXTRL_STATIC char const* NS(ComputeNodeInfo_get_arch)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN SIXTRL_STATIC char const* NS(ComputeNodeInfo_get_platform)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN SIXTRL_STATIC char const* NS(ComputeNodeInfo_get_name)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

SIXTRL_HOST_FN SIXTRL_STATIC char const* NS(ComputeNodeInfo_get_description)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info );

/* ************************************************************************** */
/* ******                                                               ***** */
/* ******             Inline functions implementation                   ***** */
/* ******                                                               ***** */
/* ************************************************************************** */

SIXTRL_INLINE NS(comp_node_id_num_t) NS(ComputeNodeId_get_platform_id)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id )
{
    return ( id != 0 ) ? id->platform_id : -1;
}

SIXTRL_INLINE NS(comp_node_id_num_t) NS(ComputeNodeId_get_device_id)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id )
{
    return ( id != 0 ) ? id->device_id : -1;
}

SIXTRL_INLINE void NS(ComputeNodeId_set_platform_id)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    NS(comp_node_id_num_t) const platform_id )
{
    SIXTRL_ASSERT( id != 0 );
    id->platform_id = platform_id;
    return;
}

SIXTRL_INLINE void NS(ComputeNodeId_set_device_id)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    NS(comp_node_id_num_t) const dev_id )
{
    SIXTRL_ASSERT( id != 0 );
    id->device_id = dev_id;
    return;
}

SIXTRL_INLINE int NS(ComputeNodeId_is_valid)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id )
{
    return ( ( id != 0 ) &&
             ( id->platform_id != -1 ) &&
             ( id->device_id   != -1 ) );
}


SIXTRL_INLINE int NS(ComputeNodeInfo_is_valid)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( ( node_info != 0 ) &&
             ( NS(ComputeNodeId_is_valid)( &node_info->id ) ) &&
             ( NS(ComputeNodeInfo_get_arch)( node_info ) != 0 ) &&
             ( strlen( NS(ComputeNodeInfo_get_arch)( node_info ) ) > 0u ) &&
             ( NS(ComputeNodeInfo_get_name)( node_info ) != 0 ) &&
             ( strlen( NS(ComputeNodeInfo_get_name)( node_info ) ) > 0u ) &&
             ( ( NS(ComputeNodeInfo_get_platform)( node_info ) == 0 ) ||
               ( strlen( NS(ComputeNodeInfo_get_platform)( node_info ) )
                   > 0u ) ) &&
             ( ( NS(ComputeNodeInfo_get_description)( node_info ) == 0 ) ||
               ( strlen( NS(ComputeNodeInfo_get_description)( node_info ) ) >
                   0u ) ) );
}

SIXTRL_INLINE NS(ComputeNodeId) NS(ComputeNodeInfo_get_id)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    SIXTRL_ASSERT( node_info != 0 );
    return node_info->id;
}

SIXTRL_INLINE NS(comp_node_id_num_t) NS(ComputeNodeInfo_get_platform_id)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != 0 )
        ? NS(ComputeNodeId_get_platform_id)( &node_info->id )
        : NS(ComputeNodeId_get_platform_id)( 0 );
}

SIXTRL_INLINE NS(comp_node_id_num_t) NS(ComputeNodeInfo_get_device_id)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != 0 )
        ? NS(ComputeNodeId_get_device_id)( &node_info->id )
        : NS(ComputeNodeId_get_device_id)( 0 );
}

SIXTRL_INLINE char const* NS(ComputeNodeInfo_get_arch)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != 0 ) ? node_info->arch : 0;
}

SIXTRL_INLINE char const* NS(ComputeNodeInfo_get_platform)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != 0 ) ? node_info->platform : 0;
}

SIXTRL_INLINE char const* NS(ComputeNodeInfo_get_name)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != 0 ) ? node_info->name : 0;
}

SIXTRL_INLINE char const* NS(ComputeNodeInfo_get_description)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return ( node_info != 0 ) ? node_info->description : 0;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_COMPUTE_ARCH_H__ */

/* end: sixtracklib/common/compute_arch.h */
