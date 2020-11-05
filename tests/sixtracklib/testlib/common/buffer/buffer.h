#ifndef SIXTRACKLIB_TESTLIB_COMMON_BUFFER_H__
#define SIXTRACKLIB_TESTLIB_COMMON_BUFFER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdio.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/internal/buffer_object_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Object);

SIXTRL_STATIC SIXTRL_FN void NS(Buffer_object_print_out_typeid)(
    NS(object_type_id_t) const type_id );

SIXTRL_STATIC SIXTRL_FN void NS(Buffer_object_print_out)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const struct NS(Object) *const
        SIXTRL_RESTRICT obj );

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print_out_typeid_ext)(
    NS(object_type_id_t) const type_id );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print_out_ext)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const struct NS(Object) *const
        SIXTRL_RESTRICT obj );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print_typeid)(
    SIXTRL_ARGPTR_DEC FILE* SIXTRL_RESTRICT fp,
    NS(object_type_id_t) const type_id );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_typeid_to_string)(
    NS(object_type_id_t) const type_id,
    char* SIXTRL_RESTRICT type_str,
    NS(buffer_size_t) const max_type_str_length );

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(Buffer_object_print)(
    SIXTRL_ARGPTR_DEC FILE* fp,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const struct NS(Object) *const
        SIXTRL_RESTRICT obj );

#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************ */
/* ************************************************************************ */

#if !defined( _GPUCODE )
    #include <stdio.h>
#endif /* !defined( _GPUCODE ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_INLINE void NS(Buffer_object_print_out_typeid)(
    NS(object_type_id_t) const type_id )
{
    NS(Buffer_object_print_typeid)( stdout, type_id );
}

SIXTRL_INLINE void NS(Buffer_object_print_out)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    const struct NS(Object) *const SIXTRL_RESTRICT obj )
{
    NS(Buffer_object_print)( stdout, obj );
}

#else  /*  defined( _GPUCODE ) */

SIXTRL_INLINE void NS(Buffer_object_print_out_typeid)(
    NS(object_type_id_t) const type_id )
{
    switch( type_id )
    {
        case NS(OBJECT_TYPE_NONE):
        {
            printf( "None" );
            break;
        }

        case NS(OBJECT_TYPE_PARTICLE):
        {
            printf( "ParticleSet" );
            break;
        }

        case NS(OBJECT_TYPE_DRIFT):
        {
            printf( "Drift" );
            break;
        }

        case NS(OBJECT_TYPE_DRIFT_EXACT):
        {
            printf( "DriftExact" );
            break;
        }

        case NS(OBJECT_TYPE_MULTIPOLE):
        {
            printf( "Multipole" );
            break;
        }

        case NS(OBJECT_TYPE_CAVITY):
        {
            printf( "Cavity" );
            break;
        }

        case NS(OBJECT_TYPE_XYSHIFT):
        {
            printf( "XYShift" );
            break;
        }

        case NS(OBJECT_TYPE_SROTATION):
        {
            printf( "SRotation" );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            printf( "BeamBeam4D" );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_6D):
        {
            printf( "BeamBeam6D" );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_MONITOR):
        {
            printf( "BeamMonitor" );
            break;
        }

        case NS(OBJECT_TYPE_LIMIT_RECT):
        {
            printf( "LimitRect" );
            break;
        }

        case NS(OBJECT_TYPE_LIMIT_ELLIPSE):
        {
            printf( "LimitEllipse" );
            break;
        }

        case NS(OBJECT_TYPE_SPACE_CHARGE_COASTING):
        {
            printf( "SpaceChargeCoasting" );
            break;
        }

        case NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED):
        {
            printf( "SpaceChargeBunched" );
            break;
        }

        case NS(OBJECT_TYPE_DIPEDGE):
        {
            printf( "DipoleEdge" );
            break;
        }

        case NS(OBJECT_TYPE_PARTICLES_ADDR):
        {
            printf( "ParticlesAddr" );
            break;
        }

        case NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM):
        {
            printf( "AssignAddressItem" );
            break;
        }

        case NS(OBJECT_TYPE_LINE):
        {
            printf( "Line" );
            break;
        }

        case NS(OBJECT_TYPE_BINARY_ARRAY):
        {
            printf( "BinaryArray" );
            break;
        }

        case NS(OBJECT_TYPE_REAL_ARRAY):
        {
            printf( "RealArray" );
            break;
        }

        case NS(OBJECT_TYPE_FLOAT32_ARRAY):
        {
            printf( "Float32Array" );
            break;
        }

        case NS(OBJECT_TYPE_UINT64_ARRAY):
        {
            printf( "UInt64Array" );
            break;
        }

        case NS(OBJECT_TYPE_INT64_ARRAY):
        {
            printf( "Int64Array" );
            break;
        }

        case NS(OBJECT_TYPE_UINT32_ARRAY):
        {
            printf( "UInt32Array" );
            break;
        }

        case NS(OBJECT_TYPE_INT32_ARRAY):
        {
            printf( "Int32Array" );
            break;
        }

        case NS(OBJECT_TYPE_UINT16_ARRAY):
        {
            printf( "UInt16Array" );
            break;
        }

        case NS(OBJECT_TYPE_INT16_ARRAY):
        {
            printf( "Int16Array" );
            break;
        }

        case NS(OBJECT_TYPE_INT8_ARRAY):
        {
            printf( "Int8Array" );
            break;
        }

        case NS(OBJECT_TYPE_CSTRING):
        {
            printf( "CString" );
            break;
        }

        case NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF):
        {
            printf( "ElemByElemConfig" );
            break;
        }

        default:
        {
            printf( "n/a" );
        }
    };
}

SIXTRL_INLINE void NS(Buffer_object_print_out)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
    const struct NS(Object) *const SIXTRL_RESTRICT obj )
{
    typedef NS(buffer_addr_t) address_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(object_type_id_t) type_id_t;

    type_id_t  const type_id = NS(Object_get_type_id)( obj );
    address_t  const addr    = NS(Object_get_begin_addr)( obj );
    buf_size_t const size    = NS(Object_get_size)( obj );

    if( ( addr != ( address_t )0u ) && ( size > ( buf_size_t )0u ) )
    {
        printf( "address = %11p, size = %6lu Bytes\r\n",
                ( void* )( uintptr_t )addr, ( unsigned long )size );

        if( type_id == NS(OBJECT_TYPE_PARTICLE) )
        {
            typedef SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) const* ptr_t;
            NS(Particles_print_out)( ( ptr_t )( uintptr_t )addr );
        }
        else if( type_id == NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) )
        {
            typedef SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
                    ptr_t;

            ptr_t config = ( ptr_t )( uintptr_t )addr;
            SIXTRL_ASSERT( config != SIXTRL_NULLPTR );

            printf( "|elem_config | order            = %16ld ;\r\n"
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
            ( int64_t )NS(ElemByElemConfig_get_num_particles_to_store)( config ),
            ( int64_t )NS(ElemByElemConfig_get_num_elements_to_store)( config ),
            ( int64_t )NS(ElemByElemConfig_get_num_turns_to_store)( config ),
            ( int64_t )NS(ElemByElemConfig_get_min_particle_id)( config ),
            ( int64_t )NS(ElemByElemConfig_get_max_particle_id)( config ),
            ( int64_t )NS(ElemByElemConfig_get_min_element_id)( config ),
            ( int64_t )NS(ElemByElemConfig_get_max_element_id)( config ),
            ( int64_t )NS(ElemByElemConfig_get_min_turn)( config ),
            ( int64_t )NS(ElemByElemConfig_get_max_turn)( config ),
            config->is_rolling,
            ( void* )( uintptr_t )config->out_store_addr );
        }
        else if( ( type_id >= NS(OBJECT_TYPE_BINARY_ARRAY) ) &&
                 ( type_id <= NS(OBJECT_TYPE_CSTRING) ) )
        {
            printf( "TODO\r\n" );
        }
        else if( type_id == NS(OBJECT_TYPE_NONE) )
        {
            printf( "None Object\r\n" );
        }
        else
        {
            NS(BeamElement_print)( fp, obj );
        }
    }
}

#endif /* !defined( _GPUCODE ) */


#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_TESTLIB_COMMON_BUFFER_H__ */

/* end: tests/sixtracklib/testlib/common/buffer.h */
