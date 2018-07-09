#include "sixtracklib/common/compute_arch.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"

extern SIXTRL_HOST_FN NS(ComputeNodeId)* NS(ComputeNodeId_preset)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id );

extern SIXTRL_HOST_FN NS(ComputeNodeInfo)* NS(ComputeNodeInfo_preset)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info );

extern SIXTRL_HOST_FN void NS(ComputeNodeInfo_free)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info );

extern SIXTRL_HOST_FN NS(ComputeNodeInfo)* NS(ComputeNodeInfo_reserve)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    SIXTRL_SIZE_T const arch_str_len,
    SIXTRL_SIZE_T const platform_str_len,
    SIXTRL_SIZE_T const name_str_len,
    SIXTRL_SIZE_T const description_str_len );

extern SIXTRL_HOST_FN int NS(ComputeNodeInfo_make)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    NS(ComputeNodeId) const id,
    const char *const SIXTRL_RESTRICT arch_str,
    const char *const SIXTRL_RESTRICT platform_str,
    const char *const SIXTRL_RESTRICT name_str,
    const char *const SIXTRL_RESTRICT description_str );



NS(ComputeNodeId)* NS(ComputeNodeId_preset)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id )
{
    if( id != 0 )
    {
        id->platform_id = -1;
        id->device_id   = -1;
    }

    return id;
}

NS(ComputeNodeInfo)* NS(ComputeNodeInfo_preset)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info )
{
    if( node_info != 0 )
    {
        NS(ComputeNodeId_preset)( &node_info->id );
        node_info->arch        = 0;
        node_info->platform    = 0;
        node_info->name        = 0;
        node_info->description = 0;
    }

    return node_info;
}

SIXTRL_HOST_FN void NS(ComputeNodeInfo_free)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info )
{
    free( ( char* )NS(ComputeNodeInfo_get_arch)( node_info ) );
    free( ( char* )NS(ComputeNodeInfo_get_platform)( node_info ) );
    free( ( char* )NS(ComputeNodeInfo_get_name)( node_info ) );
    free( ( char* )NS(ComputeNodeInfo_get_description)( node_info ) );

    NS(ComputeNodeInfo_preset)( node_info );

    return;
}

SIXTRL_HOST_FN NS(ComputeNodeInfo)* NS(ComputeNodeInfo_reserve)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    SIXTRL_SIZE_T const arch_str_len,
    SIXTRL_SIZE_T const platform_str_len,
    SIXTRL_SIZE_T const name_str_len,
    SIXTRL_SIZE_T const description_str_len )
{
    NS(ComputeNodeInfo)* ptr_result = 0;

    if( ( node_info != 0 ) && ( arch_str_len > 0u ) && ( name_str_len > 0u ) )
    {
        size_t const arch_size = sizeof( char ) * ( arch_str_len + 1u );
        char* arch_str = ( char* )malloc( arch_size );

        size_t const name_size = sizeof( char ) * ( name_str_len + 1u );
        char* name_str = ( char* )malloc( name_size );

        char* platform_str = 0;
        char* description_str = 0;

        if( platform_str_len > 0u )
        {
            size_t const platform_size =
                sizeof( char ) * ( platform_str_len + 1u );

            platform_str = ( char* )malloc( platform_size );

            if( platform_str != 0 )
            {
                memset( platform_str, ( int )'\0', platform_size );
            }
        }

        if( description_str_len > 0u )
        {
            size_t const description_size =
                sizeof( char ) * ( description_str_len + 1u );

            description_str = ( char* )malloc( description_size );

            if( description_str != 0 )
            {
                memset( description_str, ( int )'\0', description_size );
            }
        }

        if( ( arch_str != 0 ) && ( name_str != 0 ) &&
            ( ( platform_str_len    == 0u ) || ( platform_str    != 0 ) ) &&
            ( ( description_str_len == 0u ) || ( description_str != 0 ) ) )
        {
            NS(ComputeNodeInfo_free)( node_info );

            memset( arch_str, ( int )'\0', arch_size );
            memset( name_str, ( int )'\0', name_size );

            node_info->arch        = arch_str;
            node_info->name        = name_str;
            node_info->platform    = platform_str;
            node_info->description = description_str;

            ptr_result = node_info;
        }
        else
        {
            free( arch_str );
            free( name_str );
            free( platform_str );
            free( description_str );

            arch_str        = 0;
            name_str        = 0;
            platform_str    = 0;
            description_str = 0;
        }
    }

    return ptr_result;
}

SIXTRL_HOST_FN int NS(ComputeNodeInfo_make)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    NS(ComputeNodeId) const id,
    const char *const SIXTRL_RESTRICT arch_str,
    const char *const SIXTRL_RESTRICT platform_str,
    const char *const SIXTRL_RESTRICT name_str,
    const char *const SIXTRL_RESTRICT description_str )
{
    int success = -1;

    size_t const arch_len = ( arch_str != 0 ) ? strlen( arch_str ) : 0u;
    size_t const name_len = ( name_str != 0 ) ? strlen( name_str ) : 0u;

    size_t const platform_len = ( platform_str != 0 )
        ? strlen( platform_str ) : 0u;

    size_t const description_len = ( description_str != 0 )
        ? strlen( description_str ) : 0u;

    if( ( arch_len > 0u ) && ( name_len > 0u ) &&
        ( ( platform_len    > 0u ) || ( platform_str == 0 ) ) &&
        ( ( description_len > 0u ) || ( description_str == 0 ) ) &&
        ( 0 != NS(ComputeNodeInfo_reserve)(
            node_info, arch_len, name_len, platform_len, description_len ) ) )
    {
        SIXTRL_ASSERT( NS(ComputeNodeInfo_get_arch)( node_info ) != 0 );
        SIXTRL_ASSERT( NS(ComputeNodeInfo_get_name)( node_info ) != 0 );
        SIXTRL_ASSERT(
            ( ( ( NS(ComputeNodeInfo_get_platform)( node_info ) != 0 ) &&
                ( platform_len != 0u ) ) ||
              ( ( NS(ComputeNodeInfo_get_platform)( node_info ) == 0 ) &&
                ( platform_len == 0u ) ) ) );

        SIXTRL_ASSERT(
            ( ( ( NS(ComputeNodeInfo_get_description)( node_info ) != 0 ) &&
                ( description_len != 0u ) ) ||
              ( ( NS(ComputeNodeInfo_get_description)( node_info ) == 0 ) &&
                ( description_len == 0u ) ) ) );

        strncpy( ( char* )NS(ComputeNodeInfo_get_arch)( node_info ),
                 arch_str, arch_len );

        strncpy( ( char* )NS(ComputeNodeInfo_get_name)( node_info ),
                 name_str, name_len );

        if( platform_len > 0 )
        {
            strncpy( ( char* )NS(ComputeNodeInfo_get_platform)( node_info ),
                     platform_str, platform_len );
        }

        if( description_len > 0 )
        {
            strncpy( ( char* )NS(ComputeNodeInfo_get_description)( node_info ),
                     description_str, description_len );
        }

        node_info->id = id;

        success = 0;
    }

    return success;
}

/* end: sixtracklib/common/details/compute_arch.c  */
