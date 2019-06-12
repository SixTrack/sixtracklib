#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/testlib/common/particles.h"
    #include "sixtracklib/testlib/common/beam_elements/beam_elements.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

int NS(Buffer_object_typeid_to_string)(
    NS(object_type_id_t) const type_id, char* SIXTRL_RESTRICT type_str,
    NS(buffer_size_t) const max_len )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const pref_len = strlen( SIXTRL_C99_NAMESPACE_PREFIX_STR ) + 1u;

    if( ( type_str != SIXTRL_NULLPTR ) && ( max_len > pref_len ) )
    {
        buf_size_t const len = max_len - 1u;

        memset( type_str, ( int )'\0', max_len );

        success = 0;

        switch( type_id )
        {
            case NS(OBJECT_TYPE_NONE):
            {
                strncpy( type_str, "None", max_len - 1u );
                break;
            }

            case NS(OBJECT_TYPE_PARTICLE):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "Particle", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "Drift", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_DRIFT_EXACT):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "DriftExact", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_MULTIPOLE):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "Multipole", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_CAVITY):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "Cavity", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_XYSHIFT):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "XYShift", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_SROTATION):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "SRotation", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_4D):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "BeamBeam4D", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_BEAM_6D):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "BeamBeam6D", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_BEAM_MONITOR):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "BeamMonitor", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_LINE):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "Line", max_len - pref_len );
                break;
            }

            case NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF):
            {
                strncpy( type_str, SIXTRL_C99_NAMESPACE_PREFIX_STR, len );
                strncat( type_str, "ElemByElemConfig", max_len - pref_len );
                break;
            }

            default:
            {
                strncpy( type_str, "n/a", max_len - 1u );
                success = -1;
            }
        };
    }

    return success;
}

void NS(Buffer_object_print_typeid)( FILE* SIXTRL_RESTRICT fp,
    NS(object_type_id_t) const type_id )
{
    if( fp != SIXTRL_NULLPTR )
    {
        char tmp_str[ 64 ];

        if( 0 == NS(Buffer_object_typeid_to_string)(
                type_id, &tmp_str[ 0 ], 64 ) )
        {
            fprintf( fp, "Object type: %s", tmp_str );
        }
        else
        {
            fprintf( fp, "Object type: n/a" );
        }
    }

    return;
}

void NS(Buffer_object_print_out_typeid)( NS(object_type_id_t) const type_id )
{
    NS(Buffer_object_print_typeid)( stdout, type_id );
    return;
}



void NS(Buffer_object_print)( SIXTRL_ARGPTR_DEC FILE* fp,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(object_type_id_t)    type_id_t;

    type_id_t  const type_id = NS(Object_get_type_id)( obj );
    address_t  const addr = NS(Object_get_begin_addr)( obj );
    buf_size_t const size = NS(Object_get_size)( obj );

    if( ( fp != SIXTRL_NULLPTR ) && ( addr != ( address_t )0u ) &&
        ( size > ( buf_size_t )0u ) )
    {
        fprintf( fp, "address = %11p, size = %6lu Bytes\r\n",
                 ( void* )( uintptr_t )addr, ( unsigned long )size );

        if( type_id == NS(OBJECT_TYPE_PARTICLE) )
        {
            typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const* ptr_t;
            NS(Particles_print)( fp, ( ptr_t )( uintptr_t )addr );
        }
        else if( type_id == NS(OBJECT_TYPE_NONE) )
        {
            fprintf( fp, "None Object\r\n" );
        }
        else if( type_id == NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) )
        {
            typedef SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
                    ptr_t;

            ptr_t config = ( ptr_t )( uintptr_t )addr;
            SIXTRL_ASSERT( config != SIXTRL_NULLPTR );

            fprintf( fp, "|elem_config | order            = %16ld ;\r\n"
                         "             | nparts_to_store  = %16ld ;\r\n"
                         "             | nelems_to_store  = %16ld ;\r\n"
                         "             | nturns_to_store  = %16ld ;\r\n"
                         "             | min_partid       = %16ld ;\r\n"
                         "             | max_partid       = %16ld ;\r\n"
                         "             | min_elemid       = %16ld ;\r\n"
                         "             | max_elemid       = %16ld ;\r\n"
                         "             | min_turn         = %16ld ;\r\n"
                         "             | max_turn         = %16ld ;\r\n"
                         "             | is_rolling       = %16ld ;\r\n"
                         "             | out_store_addr   = %16p ;\r\n",
                ( int64_t )NS(ElemByElemConfig_get_order)( config ),
                ( int64_t )NS(ElemByElemConfig_get_num_particles_to_store)(
                    config ),
                ( int64_t )NS(ElemByElemConfig_get_num_elements_to_store)(
                    config ),
                ( int64_t )NS(ElemByElemConfig_get_num_turns_to_store)(
                    config ),
                ( int64_t )NS(ElemByElemConfig_get_min_particle_id)( config ),
                ( int64_t )NS(ElemByElemConfig_get_max_particle_id)( config ),
                ( int64_t )NS(ElemByElemConfig_get_min_element_id)( config ),
                ( int64_t )NS(ElemByElemConfig_get_max_element_id)( config ),
                ( int64_t )NS(ElemByElemConfig_get_min_turn)( config ),
                ( int64_t )NS(ElemByElemConfig_get_max_turn)( config ),
                config->is_rolling,
                ( void* )( uintptr_t )config->out_store_addr );
        }
        else
        {
            NS(BeamElement_fprint)( fp, obj );
        }
    }

    return;
}

void NS(Buffer_object_print_out)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const SIXTRL_RESTRICT obj )
{
    NS(Buffer_object_print)( stdout, obj );
    return;
}


/* end: tests/sixtracklib/testlib/common/buffer.c */
