#ifndef SIXTRACKLIB_COMMON_INTERNAL_COMPUTE_ARCH_H__
#define SIXTRACKLIB_COMMON_INTERNAL_COMPUTE_ARCH_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
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

SIXTRL_FN SIXTRL_STATIC bool operator<(
    NS(ComputeNodeId) const& lhs,
    NS(ComputeNodeId) const& rhs ) SIXTRL_NOEXCEPT;

#endif /* !defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ComputeNodeId)* NS(ComputeNodeId_preset)(
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

SIXTRL_HOST_FN SIXTRL_STATIC int NS(ComputeNodeId_compare)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT lhs,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT rhs );

SIXTRL_HOST_FN SIXTRL_STATIC bool NS(ComputeNodeId_are_equal)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT lhs,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT rhs );

#if !defined( GPUCODE )

SIXTRL_HOST_FN int NS(ComputeNodeId_to_string)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id,
    char* SIXTRL_RESTRICT str_buffer,
    SIXTRL_UINT64_T const str_buffer_capacity );

SIXTRL_HOST_FN int NS(ComputeNodeId_from_string)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    char const* SIXTRL_RESTRICT str_buffer );

#endif /* !defined( _GPUCODE ) */



typedef struct NS(ComputeNodeInfo)
{
    NS(ComputeNodeId) id;
    char*   arch;
    char*   platform;
    char*   name;
    char*   description;
}
NS(ComputeNodeInfo);

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ComputeNodeInfo)* NS(ComputeNodeInfo_preset)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ComputeNodeInfo_free)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(ComputeNodeInfo)* NS(ComputeNodeInfo_reserve)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    SIXTRL_SIZE_T const arch_str_len,
    SIXTRL_SIZE_T const platform_str_len,
    SIXTRL_SIZE_T const name_str_len,
    SIXTRL_SIZE_T const description_str_len );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(ComputeNodeInfo_make)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    NS(ComputeNodeId) const id,
    const char *const SIXTRL_RESTRICT arch_str,
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


#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ComputeNodeInfo_print)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT default_node_id );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(ComputeNodeInfo_print_out)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT default_node_id );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t) NS(ComputeNodeInfo_print_to_str)(
    char* SIXTRL_RESTRICT node_info_out_str,
    NS(arch_size_t) const node_info_out_str_capacity,
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT default_node_id );

#else /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC void NS(ComputeNodeInfo_print_out)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT default_node_id );

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************** */
/* ******                                                               ***** */
/* ******             Inline functions implementation                   ***** */
/* ******                                                               ***** */
/* ************************************************************************** */

#if defined( __cplusplus )

SIXTRL_INLINE bool operator<( NS(ComputeNodeId) const& lhs,
    NS(ComputeNodeId) const& rhs ) SIXTRL_NOEXCEPT
{
    return ( ( lhs.platform_id < rhs.platform_id ) ||
             ( ( lhs.platform_id == rhs.platform_id ) &&
               ( lhs.device_id   <  rhs.device_id   ) ) );
}

#endif /* defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

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

SIXTRL_INLINE int NS(ComputeNodeId_compare)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT lhs,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT rhs )
{
    int compare_result = -1;

    bool const lhs_is_valid = NS(ComputeNodeId_is_valid)( lhs );

    if( ( lhs_is_valid ) && ( NS(ComputeNodeId_is_valid)( rhs ) ) )
    {
        NS(comp_node_id_num_t) const lhs_platform_idx =
            NS(ComputeNodeId_get_platform_id)( lhs );

        NS(comp_node_id_num_t) const rhs_platform_idx =
            NS(ComputeNodeId_get_platform_id)( rhs );

        if( lhs_platform_idx == rhs_platform_idx )
        {
            NS(comp_node_id_num_t) const lhs_device_idx =
                NS(ComputeNodeId_get_device_id)( lhs );

            NS(comp_node_id_num_t) const rhs_device_idx =
                NS(ComputeNodeId_get_device_id)( rhs );

            if( lhs_device_idx == rhs_device_idx )
            {
                compare_result = 0;
            }
            else if( lhs_device_idx > rhs_device_idx )
            {
                compare_result = +1;
            }
        }
        else if( lhs_platform_idx > rhs_platform_idx )
        {
            compare_result = +1;
        }
    }
    else if( lhs_is_valid )
    {
        compare_result = +1;
    }

    return compare_result;
}

SIXTRL_INLINE bool NS(ComputeNodeId_are_equal)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT lhs,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT rhs )
{
    return ( NS(ComputeNodeId_compare)( lhs, rhs ) == 0 );
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

#endif /* SIXTRACKLIB_COMMON_INTERNAL_COMPUTE_ARCH_H__ */

/* end: sixtracklib/common/context/compute_arch.h */
