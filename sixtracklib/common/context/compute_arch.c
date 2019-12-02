#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/context/compute_arch.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <stdio.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

NS(ComputeNodeId)* NS(ComputeNodeId_allocate_array)(
    NS(arch_size_t) const num_nodes )
{
    NS(ComputeNodeId)* array = SIXTRL_NULLPTR;

    if( num_nodes > ( NS(arch_size_t) )0u )
    {
        array = ( NS(ComputeNodeId)* )malloc(
            sizeof( NS(ComputeNodeId) ) * num_nodes );

        if( array != SIXTRL_NULLPTR )
        {
            NS(arch_size_t) ii = ( NS(arch_size_t) )0u;

            for( ; ii < num_nodes ; ++ii )
            {
                NS(ComputeNodeId_preset)( &array[ ii ] );
            }
        }
    }

    return array;
}

void NS(ComputeNodeId_free_array)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT array_begin )
{
    if( array_begin != SIXTRL_NULLPTR )
    {
        free( array_begin );
    }
}

NS(comp_node_id_num_t) NS(ComputeNodeId_get_platform_id_ext)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id )
{
    return NS(ComputeNodeId_get_platform_id)( id );
}

NS(comp_node_id_num_t) NS(ComputeNodeId_get_device_id_ext)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id )
{
    return NS(ComputeNodeId_get_device_id)( id );
}

void NS(ComputeNodeId_set_platform_id_ext)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    NS(comp_node_id_num_t) const platform_id )
{
    NS(ComputeNodeId_set_platform_id)( id, platform_id );
}

void NS(ComputeNodeId_set_device_id_ext)(
    NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    NS(comp_node_id_num_t) const device_id )
{
    NS(ComputeNodeId_set_device_id)( id, device_id );
}

bool NS(ComputeNodeId_is_valid_ext)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT id )
{
    return NS(ComputeNodeId_is_valid)( id );
}

int NS(ComputeNodeId_compare_ext)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT lhs,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT rhs )
{
    return NS(ComputeNodeId_compare)( lhs, rhs );
}

bool NS(ComputeNodeId_are_equal_ext)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT lhs,
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT rhs )
{
    return NS(ComputeNodeId_are_equal)( lhs, rhs );
}

#if !defined( GPUCODE )

int NS(ComputeNodeId_from_string)( NS(ComputeNodeId)* SIXTRL_RESTRICT id,
    char const* SIXTRL_RESTRICT str_buffer )
{
    int success = -1;

    if( ( str_buffer != SIXTRL_NULLPTR ) &&
        ( strlen( str_buffer ) > 0u ) &&
        ( id != SIXTRL_NULLPTR ) )
    {
        int temp_platform_idx = -1;
        int temp_device_idx   = -1;

        int const ret = sscanf( str_buffer, "%d.%d",
                                &temp_platform_idx, &temp_device_idx );

        if( ( ret == 2 ) && ( temp_platform_idx >= 0 ) &&
            ( temp_device_idx >= 0 ) )
        {
            NS(ComputeNodeId_set_platform_id)( id, temp_platform_idx );
            NS(ComputeNodeId_set_device_id)( id, temp_device_idx );

            success = 0;
        }
    }

    return success;
}

int NS(ComputeNodeId_to_string)(
    const NS(ComputeNodeId) *const SIXTRL_RESTRICT node_id,
    char* SIXTRL_RESTRICT out_string_begin,
    NS(arch_size_t) const out_string_capacity )
{
    int success = -1;

    if( ( node_id != SIXTRL_NULLPTR ) &&
        ( NS(ComputeNodeId_is_valid)( node_id ) ) &&
        ( out_string_begin != SIXTRL_NULLPTR ) &&
        ( out_string_capacity > ( NS(arch_size_t) )0u ) )
    {
        int ret = 0;
        memset( out_string_begin, ( int )'\0', out_string_capacity );

        SIXTRL_ASSERT( NS(ComputeNodeId_get_platform_id)( node_id ) >= 0 );
        SIXTRL_ASSERT( NS(ComputeNodeId_get_device_id)( node_id ) >= 0 );

        ret = snprintf( out_string_begin, out_string_capacity, "%u.%u",
                ( unsigned )NS(ComputeNodeId_get_platform_id)( node_id ),
                ( unsigned )NS(ComputeNodeId_get_device_id)( node_id ) );

        if( ( ret >= 0 ) && ( out_string_capacity > ( NS(arch_size_t) )ret ) )
        {
            success = 0;
        }
    }

    return success;
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

NS(ComputeNodeInfo)* NS(ComputeNodeInfo_preset_ext)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_preset)( node_info );
}

bool NS(ComputeNodeInfo_is_valid_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_is_valid)( node_info );
}

NS(ComputeNodeId) NS(ComputeNodeInfo_get_id_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_get_id)( node_info );
}

NS(comp_node_id_num_t) NS(ComputeNodeInfo_get_platform_id_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_get_platform_id)( node_info );
}

NS(comp_node_id_num_t) NS(ComputeNodeInfo_get_device_id_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_get_device_id)( node_info );
}

char const* NS(ComputeNodeInfo_get_arch_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_get_arch)( node_info );
}

char const* NS(ComputeNodeInfo_get_platform_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_get_platform)( node_info );
}

char const* NS(ComputeNodeInfo_get_name_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_get_name)( node_info );
}

char const* NS(ComputeNodeInfo_get_description_ext)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info )
{
    return NS(ComputeNodeInfo_get_description)( node_info );
}

#if !defined( _GPUCODE )

void NS(ComputeNodeInfo_free)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info )
{
    free( ( char* )NS(ComputeNodeInfo_get_arch)( node_info ) );
    free( ( char* )NS(ComputeNodeInfo_get_platform)( node_info ) );
    free( ( char* )NS(ComputeNodeInfo_get_name)( node_info ) );
    free( ( char* )NS(ComputeNodeInfo_get_description)( node_info ) );

    NS(ComputeNodeInfo_preset)( node_info );
}

void NS(ComputeNodeInfo_delete)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info )
{
    NS(ComputeNodeInfo_free)( node_info );
    free( node_info );
}

NS(ComputeNodeInfo)* NS(ComputeNodeInfo_reserve)(
    NS(ComputeNodeInfo)* SIXTRL_RESTRICT node_info,
    SIXTRL_SIZE_T const arch_str_len,
    SIXTRL_SIZE_T const platform_str_len,
    SIXTRL_SIZE_T const name_str_len,
    SIXTRL_SIZE_T const description_str_len )
{
    NS(ComputeNodeInfo)* ptr_result = SIXTRL_NULLPTR;

    if( ( node_info != SIXTRL_NULLPTR ) &&
        ( arch_str_len > 0u ) && ( name_str_len > 0u ) )
    {
        size_t const arch_size = sizeof( char ) * ( arch_str_len + 1u );
        char* arch_str = ( char* )malloc( arch_size );

        size_t const name_size = sizeof( char ) * ( name_str_len + 1u );
        char* name_str = ( char* )malloc( name_size );

        char* platform_str = SIXTRL_NULLPTR;
        char* description_str = SIXTRL_NULLPTR;

        if( platform_str_len > 0u )
        {
            size_t const platform_size =
                sizeof( char ) * ( platform_str_len + 1u );

            platform_str = ( char* )malloc( platform_size );

            if( platform_str != SIXTRL_NULLPTR )
            {
                memset( platform_str, ( int )'\0', platform_size );
            }
        }

        if( description_str_len > 0u )
        {
            size_t const description_size =
                sizeof( char ) * ( description_str_len + 1u );

            description_str = ( char* )malloc( description_size );

            if( description_str != SIXTRL_NULLPTR )
            {
                memset( description_str, ( int )'\0', description_size );
            }
        }

        if( ( arch_str != 0 ) && ( name_str != SIXTRL_NULLPTR ) &&
            ( ( platform_str_len    == 0u ) ||
              ( platform_str    != SIXTRL_NULLPTR ) ) &&
            ( ( description_str_len == 0u ) ||
              ( description_str != SIXTRL_NULLPTR ) ) )
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

            arch_str        = SIXTRL_NULLPTR;
            name_str        = SIXTRL_NULLPTR;
            platform_str    = SIXTRL_NULLPTR;
            description_str = SIXTRL_NULLPTR;
        }
    }

    return ptr_result;
}

int NS(ComputeNodeInfo_make)(
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



void NS(ComputeNodeInfo_print)(
    FILE* SIXTRL_RESTRICT fp,
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info,
    const NS(ComputeNodeId)   *const SIXTRL_RESTRICT default_node_id )
{
    if( ( fp != SIXTRL_NULLPTR ) && ( node_info != SIXTRL_NULLPTR ) )
    {
        NS(ComputeNodeId)const nid = NS(ComputeNodeInfo_get_id)( node_info );

        char id_str[ 16 ] =
        {
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',
            '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0'
        };

        NS(ComputeNodeId_to_string)( &nid, &id_str[ 0 ], 16 );

        fprintf( fp, "Device ID Str  : %-10s", id_str );

        if( ( default_node_id != SIXTRL_NULLPTR ) &&
            ( NS(ComputeNodeId_are_equal)( &nid, default_node_id ) ) )
        {
            fprintf( fp, " [DEFAULT]" );
        }

        fprintf( fp, "\r\nArchitecture   : " );

        if( NS(ComputeNodeInfo_get_arch)( node_info ) != SIXTRL_NULLPTR )
        {
             fprintf( fp, "%-10s", NS(ComputeNodeInfo_get_arch)( node_info ) );
        }
        else
        {
            fprintf( fp, "n/a" );
        }

        fprintf( fp, "\r\nPlatform       : " );

        if( NS(ComputeNodeInfo_get_platform)( node_info ) != SIXTRL_NULLPTR )
        {
            fprintf( fp, "%-10s", NS(ComputeNodeInfo_get_platform)( node_info ) );
        }
        else
        {
            fprintf( fp, "n/a" );
        }

        fprintf( fp, "\r\nName           : " );

        if( NS(ComputeNodeInfo_get_name)( node_info ) != SIXTRL_NULLPTR )
        {
            fprintf( fp, "%-10s", NS(ComputeNodeInfo_get_name)( node_info ) );
        }
        else
        {
            fprintf( fp, "n/a" );
        }

        fprintf( fp, "\r\n" );
    }
}

void NS(ComputeNodeInfo_print_out)(
    const NS(ComputeNodeInfo) *const SIXTRL_RESTRICT node_info,
    const NS(ComputeNodeId)   *const SIXTRL_RESTRICT default_node_id )
{
    NS(ComputeNodeInfo_print)( stdout, node_info, default_node_id );
}

#endif /* !defined( _GPUCODE ) */

/* end: sixtracklib/common/internal/compute_arch.c  */
