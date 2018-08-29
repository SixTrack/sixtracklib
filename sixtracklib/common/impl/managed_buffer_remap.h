#ifndef SIXTRL_COMMON_IMPL_MANAGED_BUFFER_REMAP_H__
#define SIXTRL_COMMON_IMPL_MANAGED_BUFFER_REMAP_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_diff_t)
    NS(ManagedBuffer_get_limit_offset_max)( void );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_diff_t)
    NS(ManagedBuffer_get_limit_offset_min)( void );

SIXTRL_FN SIXTRL_STATIC bool NS(ManagedBuffer_check_addr_arithmetic)(
    NS(buffer_addr_t) const addr,
    NS(buffer_addr_diff_t) const offset,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_addr_t) NS(ManagedBuffer_perform_addr_shift)(
    NS(buffer_addr_t) const addr,
    NS(buffer_addr_diff_t) const offset,
    NS(buffer_size_t) const slot_size );

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC int
NS(ManagedBuffer_get_addr_offset)( SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t)*
        SIXTRL_RESTRICT ptr_to_addr_offset,
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_remap_header)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_remap_section_slots)(
    SIXTRL_ARGPTR_DEC unsigned char*  SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_remap_section_objects)(
    SIXTRL_ARGPTR_DEC unsigned char*      SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_remap_section_dataptrs)(
    SIXTRL_ARGPTR_DEC unsigned char*      SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_remap_section_garbage)(
    SIXTRL_ARGPTR_DEC unsigned char*      SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(ManagedBuffer_remap)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= *
 * ======== INLINE IMPLEMENTATION                                            *
 * ========================================================================= */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/impl/buffer_object.h"
    #include "sixtracklib/common/impl/managed_buffer_minimal.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_addr_diff_t) NS(ManagedBuffer_get_limit_offset_max)()
{
    #if defined( _GPUCODE )
         #if defined( __OPENCL_VERSION__ ) &&
             defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) &&
             ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) >=
                           sizeof( ptr_to_raw_t ) );

            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MAX = LONG_MAX;

        #elif defined( __CUDACC__ ) &&
              defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) &&
              ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

            SIXTRL_ASSERT( sizeof( addr_diff_t ) >= sizeof( long long int ) );
            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MAX = NPP_MAX_64S;

        #endif /* defined( __OPENCL_VERSION__ ) */
    #elif defined( __cplusplus )
        using addr_diff_t = NS(buffer_addr_diff_t);

        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::digits >= 63u );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_signed  );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_integer );

        SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MAX =
             std::numeric_limits< addr_diff_t >::max();

    #else
        SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                       sizeof( long long int ) );

        SIXTRL_STATIC_VAR NS(buffer_addr_diff_t) const
            LIMIT_OFFSET_MAX = ( NS(buffer_addr_diff_t) )LLONG_MAX;

    #endif /* defined( _GPUCODE ) */

    return LIMIT_OFFSET_MAX;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_diff_t)
    NS(ManagedBuffer_get_limit_offset_min)( void )
{
    #if defined( _GPUCODE )
         #if defined( __OPENCL_VERSION__ ) &&
             defined( SIXTRACKLIB_ENABLE_MODULE_OPENCL ) &&
             ( SIXTRACKLIB_ENABLE_MODULE_OPENCL == 1 )

            SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) >=
                           sizeof( ptr_to_raw_t ) );

            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MIN = LONG_MIN;

        #elif defined( __CUDACC__ ) &&
              defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) &&
              ( SIXTRACKLIB_ENABLE_MODULE_CUDA == 1 )

            SIXTRL_ASSERT( sizeof( addr_diff_t ) >= sizeof( long long int ) );
            SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MIN = NPP_MIN_64S;

        #endif /* defined( __OPENCL_VERSION__ ) */
    #elif defined( __cplusplus )
        using addr_diff_t = NS(buffer_addr_diff_t);

        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::digits >= 63u );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_signed  );
        SIXTRL_ASSERT( std::numeric_limits< addr_diff_t >::is_integer );

        SIXTRL_STATIC_VAR addr_diff_t const LIMIT_OFFSET_MIN =
             std::numeric_limits< addr_diff_t >::min();

    #else
        SIXTRL_ASSERT( sizeof( NS(buffer_addr_diff_t) ) >=
                       sizeof( long long int ) );

        SIXTRL_STATIC_VAR NS(buffer_addr_diff_t) const
            LIMIT_OFFSET_MIN = ( NS(buffer_addr_diff_t) )LLONG_MIN;

    #endif /* defined( _GPUCODE ) */

    return LIMIT_OFFSET_MIN;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE bool NS(ManagedBuffer_check_addr_arithmetic)(
    NS(buffer_addr_t) const addr,
    NS(buffer_addr_diff_t) const offset, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;

    SIXTRL_ASSERT( sizeof( address_t ) == sizeof( addr_diff_t) );
    SIXTRL_ASSERT( sizeof( address_t ) >= 8u );
    SIXTRL_ASSERT( slot_size > 0u );

    addr_diff_t const LIMIT_OFFSET_MAX =
        NS(ManagedBuffer_get_limit_offset_max)();

    return (
        ( addr != ( address_t )0u ) && ( ( addr % slot_size ) == 0u ) &&
        ( ( ( offset >= 0 ) &&
            ( ( ( ( address_t )offset ) % slot_size ) == 0u ) &&
            ( ( LIMIT_OFFSET_MAX - ( address_t )offset ) >= addr ) ) ||
          ( ( offset <  0 ) &&
            ( ( ( ( address_t )-offset ) % slot_size ) == 0u ) &&
            ( addr >= ( address_t )-offset ) ) ) );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_INLINE NS(buffer_addr_t) NS(ManagedBuffer_perform_addr_shift)(
    NS(buffer_addr_t) const addr, NS(buffer_addr_diff_t) const offset,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( NS(ManagedBuffer_check_addr_arithmetic)(
        addr, offset, slot_size) );

    #if !defined( NDEBUG )
    ( void )slot_size;
    #endif /* !defined( NDEBUG ) */

    return addr + offset;
}

/* ========================================================================= */

SIXTRL_INLINE int NS(ManagedBuffer_get_addr_offset)(
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t)* SIXTRL_RESTRICT ptr_addr_offset,
    SIXTRL_ARGPTR_DEC unsigned char const* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)       buf_size_t;
    typedef NS(buffer_addr_t)       address_t;
    typedef NS(buffer_addr_diff_t)  addr_diff_t;
    typedef address_t*              ptr_to_addr_t;

    int success = -1;

    if( ( begin  != SIXTRL_NULLPTR ) &&
        ( slot_size > ( buf_size_t )0u ) &&
        ( ( ( ( uintptr_t )begin ) % slot_size ) == 0u ) )
    {
        addr_diff_t addr_offset = ( addr_diff_t )0;

        address_t const stored_base_addr =
            ( begin != SIXTRL_NULLPTR )
                ? *( ( ptr_to_addr_t )begin ) : ( address_t )0u;

        address_t const base_address =
            ( address_t )( uintptr_t )begin;

        addr_offset = ( base_address >= stored_base_addr )
            ?  ( addr_diff_t )( base_address - stored_base_addr )
            : -( addr_diff_t )( stored_base_addr - base_address );

        if(  ptr_addr_offset != SIXTRL_NULLPTR )
        {
            *ptr_addr_offset = addr_offset;
            success = 0;
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ManagedBuffer_remap_header)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                   buf_size_t;
    typedef NS(buffer_addr_diff_t)              addr_diff_t;
    typedef NS(buffer_addr_t)                   address_t;
    typedef SIXTRL_ARGPTR_DEC address_t*        ptr_to_addr_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t  const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR addr_diff_t const ZERO_DIFF = ( addr_diff_t )0u;

    if( ( begin != SIXTRL_NULLPTR ) && ( offsets != SIXTRL_NULLPTR ) &&
        ( slot_size > ZERO_SIZE ) )
    {
        addr_diff_t base_addr_offset = ( addr_diff_t )0u;

        addr_diff_t const slots_addr_offset    = offsets[ 1 ];
        addr_diff_t const objs_addr_offset     = offsets[ 2 ];
        addr_diff_t const dataptrs_addr_offset = offsets[ 3 ];
        addr_diff_t const garbage_addr_offset  = offsets[ 4 ];

        success = NS(ManagedBuffer_get_addr_offset)(
            &base_addr_offset, begin, slot_size );

        if( ( success != 0 ) ||
            ( base_addr_offset != offsets[ 0 ] ) )
        {
            return success;
        }

        SIXTRL_ASSERT( success == 0 );
        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );

        if( ( base_addr_offset     != ZERO_DIFF ) ||
            ( slots_addr_offset    != ZERO_DIFF ) ||
            ( objs_addr_offset     != ZERO_DIFF ) ||
            ( dataptrs_addr_offset != ZERO_DIFF ) ||
            ( garbage_addr_offset  != ZERO_DIFF ) )
        {
            SIXTRL_STATIC_VAR buf_size_t const BASE_ID     = ( buf_size_t )0u;
            SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID    = ( buf_size_t )3u;
            SIXTRL_STATIC_VAR buf_size_t const OBJECTS_ID  = ( buf_size_t )4u;
            SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID = ( buf_size_t )5u;
            SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID  = ( buf_size_t )6u;

            ptr_to_addr_t header = ( ptr_to_addr_t )begin;

            address_t const remap_base_addr =
                NS(ManagedBuffer_perform_addr_shift)(
                    header[ BASE_ID ], base_addr_offset, slot_size );

            if( ( remap_base_addr != ( address_t )0u ) &&
                ( remap_base_addr == ( address_t )( uintptr_t )begin ) )
            {
                #if !defined( NDEBUG )
                SIXTRL_STATIC_VAR address_t const ZERO_ADDR = ( address_t )0u;
                #endif /* !defined( NDEBUG ) */

                address_t const remap_slots_begin_addr =
                    NS(ManagedBuffer_perform_addr_shift)( header[ SLOTS_ID ],
                        slots_addr_offset, slot_size );

                address_t const remap_objs_begin_addr =
                    NS(ManagedBuffer_perform_addr_shift)( header[ OBJECTS_ID ],
                        objs_addr_offset, slot_size );

                address_t const remap_dataptrs_begin_addr =
                    NS(ManagedBuffer_perform_addr_shift)( header[ DATAPTRS_ID ],
                        dataptrs_addr_offset, slot_size );

                address_t const remap_garbage_begin_addr =
                    NS(ManagedBuffer_perform_addr_shift)( header[ GARBAGE_ID ],
                        garbage_addr_offset, slot_size );

                #if !defined( NDEBUG )
                SIXTRL_ASSERT( header[ GARBAGE_ID  ] > header[ DATAPTRS_ID ] );
                SIXTRL_ASSERT( header[ DATAPTRS_ID ] > header[ OBJECTS_ID  ] );
                SIXTRL_ASSERT( header[ OBJECTS_ID  ] > header[ SLOTS_ID    ] );
                SIXTRL_ASSERT( header[ SLOTS_ID    ] > header[ BASE_ID     ] );

                SIXTRL_ASSERT( remap_slots_begin_addr    != ZERO_ADDR );
                SIXTRL_ASSERT( remap_objs_begin_addr     != ZERO_ADDR );
                SIXTRL_ASSERT( remap_dataptrs_begin_addr != ZERO_ADDR );
                SIXTRL_ASSERT( remap_garbage_begin_addr  != ZERO_ADDR );

                SIXTRL_ASSERT( remap_slots_begin_addr > remap_base_addr );
                SIXTRL_ASSERT( remap_objs_begin_addr  >
                               remap_slots_begin_addr );

                SIXTRL_ASSERT( remap_dataptrs_begin_addr >
                               remap_objs_begin_addr );

                SIXTRL_ASSERT( remap_garbage_begin_addr  >
                               remap_dataptrs_begin_addr );

                #endif /* !defined( NDEBUG ) */

                if( ( remap_base_addr           != ZERO_ADDR ) &&
                    ( remap_slots_begin_addr    != ZERO_ADDR ) &&
                    ( remap_objs_begin_addr     != ZERO_ADDR ) &&
                    ( remap_dataptrs_begin_addr != ZERO_ADDR ) &&
                    ( remap_garbage_begin_addr  != ZERO_ADDR ) )
                {
                    header[ BASE_ID     ] = remap_base_addr;
                    header[ SLOTS_ID    ] = remap_slots_begin_addr;
                    header[ OBJECTS_ID  ] = remap_objs_begin_addr;
                    header[ DATAPTRS_ID ] = remap_dataptrs_begin_addr;
                    header[ GARBAGE_ID  ] = remap_garbage_begin_addr;
                }
                else
                {
                    success = -1;
                }
            }
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ManagedBuffer_remap_section_slots)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)               buf_size_t;
    typedef NS(buffer_addr_t)               address_t;
    typedef SIXTRL_ARGPTR_DEC address_t*    ptr_to_addr_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    if( ( begin != SIXTRL_NULLPTR ) &&
        ( offsets != SIXTRL_NULLPTR ) && ( slot_size != ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )begin ) % slot_size ) == 0u ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const SECTION_ID = 3u;

        ptr_to_addr_t ptr_section_begin = ( ptr_to_addr_t )(
            NS(ManagedBuffer_get_const_ptr_to_section)( begin,
                SECTION_ID, slot_size ) );

        if( ptr_section_begin != SIXTRL_NULLPTR )
        {
            buf_size_t const slots_capacity  =  ptr_section_begin[ 0 ];
            buf_size_t const num_entities    =  ptr_section_begin[ 1 ];
            buf_size_t const entity_size     =
                NS(ManagedBuffer_get_section_entity_size)(
                    begin, SECTION_ID, slot_size );

            success = ( slots_capacity >= ( num_entities * entity_size ) )
                      ? 0 : -1;
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ManagedBuffer_remap_section_objects)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)               buf_size_t;
    typedef NS(buffer_addr_t)               address_t;
    typedef NS(buffer_addr_diff_t)          addr_diff_t;
    typedef SIXTRL_ARGPTR_DEC address_t*    ptr_to_addr_t;
    typedef struct NS(Object)               object_t;
    typedef SIXTRL_ARGPTR_DEC object_t*     ptr_to_object_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const ZERO_ADDR = ( address_t  )0u;

    if( ( begin != SIXTRL_NULLPTR ) &&
        ( offsets != SIXTRL_NULLPTR ) && ( slot_size != ZERO_SIZE ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID = ( buf_size_t )3u;
        SIXTRL_STATIC_VAR buf_size_t const OBJS_ID  = ( buf_size_t )4u;

        ptr_to_addr_t ptr_section_begin = ( ptr_to_addr_t )(
            NS(ManagedBuffer_get_ptr_to_section)(
                begin, OBJS_ID, slot_size ) );

        buf_size_t const objs_max_size =
            NS(ManagedBuffer_get_section_max_size)(
                begin, OBJS_ID, slot_size );

        buf_size_t const objs_size =
            NS(ManagedBuffer_get_section_size)( begin, OBJS_ID, slot_size );

        ptr_to_object_t ptr_objects_begin = (ptr_to_object_t
            )NS(ManagedBuffer_get_ptr_to_section_data)(
                begin, OBJS_ID, slot_size );

        buf_size_t const num_objects =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, OBJS_ID, slot_size );

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );

        success = ( ( ptr_section_begin != SIXTRL_NULLPTR ) &&
                    ( objs_max_size >= objs_size          ) ) ? 0 : -1;

        if( ( success == 0 ) && ( num_objects > ZERO_SIZE ) )
        {
            addr_diff_t const slots_addr_offset = offsets[ 1 ];

            address_t const slots_section_begin_addr = ( address_t )( uintptr_t
                )NS(ManagedBuffer_get_ptr_to_section_data)(
                    begin, SLOTS_ID, slot_size );

            address_t const slots_section_end_addr = ( address_t )( uintptr_t
                )NS(ManagedBuffer_get_ptr_to_section_end)(
                    begin, SLOTS_ID, slot_size );

            address_t min_valid_obj_addr = slots_section_begin_addr;

            ptr_to_object_t obj_it   = ptr_objects_begin;
            ptr_to_object_t obj_end  = ptr_objects_begin + num_objects;

            SIXTRL_ASSERT( ptr_objects_begin != SIXTRL_NULLPTR );

            for( ; obj_it != obj_end ; ++obj_it )
            {
                buf_size_t const obj_size = NS(Object_get_size)( obj_it );

                buf_size_t const obj_offset =
                    NS(ManagedBuffer_get_slot_based_length)(
                        obj_size, slot_size );

                address_t const obj_begin_addr =
                    NS(Object_get_begin_addr)( obj_it );

                address_t const remapped_obj_begin_addr =
                    NS(ManagedBuffer_perform_addr_shift)(
                        obj_begin_addr, slots_addr_offset, slot_size );

                SIXTRL_ASSERT( ( min_valid_obj_addr % slot_size ) == 0u );
                SIXTRL_ASSERT(   min_valid_obj_addr <= slots_section_end_addr );

                SIXTRL_ASSERT(   remapped_obj_begin_addr != ZERO_ADDR );
                SIXTRL_ASSERT( ( remapped_obj_begin_addr % slot_size ) == 0u );

                if( ( remapped_obj_begin_addr < slots_section_begin_addr ) ||
                    ( remapped_obj_begin_addr > slots_section_end_addr   ) ||
                    ( remapped_obj_begin_addr < min_valid_obj_addr       ) )
                {
                    success = -1;
                    break;
                }

                min_valid_obj_addr = remapped_obj_begin_addr + obj_offset;
                NS(Object_set_begin_addr)( obj_it, remapped_obj_begin_addr );
            }
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ManagedBuffer_remap_section_dataptrs)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)            buf_size_t;
    typedef NS(buffer_addr_t)            address_t;
    typedef NS(buffer_addr_diff_t)       addr_diff_t;
    typedef SIXTRL_ARGPTR_DEC address_t* ptr_to_addr_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t  )0u;

    if( ( begin != SIXTRL_NULLPTR ) &&
        ( offsets != SIXTRL_NULLPTR ) && ( slot_size != ZERO_SIZE ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID    = ( buf_size_t )3u;
        SIXTRL_STATIC_VAR buf_size_t const DATAPTRS_ID = ( buf_size_t )5u;

        ptr_to_addr_t ptr_section_begin = ( ptr_to_addr_t )(
            NS(ManagedBuffer_get_ptr_to_section_data)(
                begin, DATAPTRS_ID, slot_size ) );

        buf_size_t const num_dataptrs =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, DATAPTRS_ID, slot_size );

        buf_size_t const dataptrs_size =
            NS(ManagedBuffer_get_section_size)(
                begin, DATAPTRS_ID, slot_size );

        buf_size_t const max_dataptrs_size =
            NS(ManagedBuffer_get_section_max_size)(
                begin, DATAPTRS_ID, slot_size );

        SIXTRL_ASSERT( ( ( ( uintptr_t )begin ) % slot_size ) == 0u );

        success = ( ( ptr_section_begin != SIXTRL_NULLPTR ) &&
                    ( max_dataptrs_size >= dataptrs_size  ) ) ? 0 : -1;

        if( ( success == 0 ) && ( num_dataptrs > ZERO_SIZE ) )
        {
            addr_diff_t const slots_addr_offset = offsets[ 1 ];

            address_t const slots_section_begin_addr = ( address_t )( uintptr_t
                )NS(ManagedBuffer_get_ptr_to_section_data)(
                    begin, SLOTS_ID, slot_size );

            address_t const slots_section_end_addr = ( address_t )( uintptr_t
                )NS(ManagedBuffer_get_ptr_to_section_end)(
                    begin, SLOTS_ID, slot_size );

            ptr_to_addr_t dataptr_it  = ptr_section_begin;
            ptr_to_addr_t dataptr_end = dataptr_it + num_dataptrs;

            SIXTRL_ASSERT( ptr_section_begin != SIXTRL_NULLPTR );
            SIXTRL_ASSERT( slots_section_begin_addr != ZERO_ADDR );

            for( ; dataptr_it != dataptr_end ; ++dataptr_it )
            {
                address_t const slot_ptr_addr = *dataptr_it;

                address_t const remap_slot_ptr_addr =
                    NS(ManagedBuffer_perform_addr_shift)( slot_ptr_addr,
                        slots_addr_offset, slot_size );

                ptr_to_addr_t slot_ptr =
                    ( ptr_to_addr_t )( uintptr_t )remap_slot_ptr_addr;

                address_t const remap_slot_addr =
                    ( slot_ptr != SIXTRL_NULLPTR )
                    ? ( NS(ManagedBuffer_perform_addr_shift)( *slot_ptr,
                            slots_addr_offset, slot_size ) )
                    : ( address_t )0u;

                if( ( remap_slot_ptr_addr != ZERO_ADDR ) &&
                    ( remap_slot_addr     != ZERO_ADDR ) &&
                    ( remap_slot_ptr_addr >= slots_section_begin_addr ) &&
                    ( remap_slot_ptr_addr <  slots_section_end_addr   ) &&
                    ( remap_slot_addr     >= slots_section_begin_addr ) &&
                    ( remap_slot_addr     <  slots_section_end_addr   ) )
                {
                    *slot_ptr   = remap_slot_addr;
                    *dataptr_it = remap_slot_ptr_addr;
                }
                else
                {
                    success = -1;
                    break;
                }
            }
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ManagedBuffer_remap_section_garbage)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    SIXTRL_ARGPTR_DEC NS(buffer_addr_diff_t) const* SIXTRL_RESTRICT offsets,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t)                   buf_size_t;
    typedef NS(buffer_addr_t)                   address_t;
    typedef NS(buffer_addr_diff_t)              addr_diff_t;
    typedef SIXTRL_ARGPTR_DEC  address_t*       ptr_to_addr_t;
    typedef NS(BufferGarbage)                   garbage_range_t;
    typedef SIXTRL_ARGPTR_DEC garbage_range_t*  ptr_to_garbage_t;

    int success = -1;

    SIXTRL_STATIC_VAR buf_size_t const ZERO_SIZE = ( buf_size_t )0u;

    if( ( begin != SIXTRL_NULLPTR ) &&
        ( offsets != SIXTRL_NULLPTR ) && ( slot_size != ZERO_SIZE ) &&
        ( ( ( ( uintptr_t )begin ) % slot_size ) == 0u ) )
    {
        SIXTRL_STATIC_VAR buf_size_t const SLOTS_ID   = ( buf_size_t )3u;
        SIXTRL_STATIC_VAR buf_size_t const GARBAGE_ID = ( buf_size_t )6u;

        ptr_to_addr_t ptr_section_begin = ( ptr_to_addr_t )(
            NS(ManagedBuffer_get_ptr_to_section_data)(
                begin, GARBAGE_ID, slot_size ) );

        buf_size_t const num_garbage_ranges =
            NS(ManagedBuffer_get_section_num_entities)(
                begin, GARBAGE_ID, slot_size );

        buf_size_t const garbage_ranges_size =
            NS(ManagedBuffer_get_section_size)( begin, GARBAGE_ID, slot_size );

        buf_size_t const max_garbage_range_size =
            NS(ManagedBuffer_get_section_max_size)(
                begin, GARBAGE_ID, slot_size );

        success = ( ( ptr_section_begin != SIXTRL_NULLPTR ) &&
                    ( max_garbage_range_size >= garbage_ranges_size  ) )
                  ? 0 : -1;

        if( ( success == 0 ) && ( num_garbage_ranges > ZERO_SIZE ) )
        {
            addr_diff_t const slots_addr_offset = offsets[ 1 ];

            address_t const slots_section_begin_addr = ( address_t )(
                NS(ManagedBuffer_get_ptr_to_section_data)(
                    begin, SLOTS_ID, slot_size ) );

            address_t const slots_section_end_addr = ( address_t )(
                NS(ManagedBuffer_get_ptr_to_section_end)(
                    begin, SLOTS_ID, slot_size ) );

            ptr_to_garbage_t it  = ( ptr_to_garbage_t )ptr_section_begin;
            ptr_to_garbage_t end = it + num_garbage_ranges;

            SIXTRL_ASSERT( ptr_section_begin != SIXTRL_NULLPTR );

            for( ; it != end ; ++it )
            {
                address_t const garbage_begin_addr =
                    NS(BufferGarbage_get_begin_addr)( it );

                address_t const remap_slot_addr =
                    NS(ManagedBuffer_perform_addr_shift)( garbage_begin_addr,
                        slots_addr_offset, slot_size );

                if( ( remap_slot_addr >= slots_section_begin_addr ) &&
                    ( remap_slot_addr <  slots_section_end_addr   ) )
                {
                    NS(BufferGarbage_set_begin_addr)( it, remap_slot_addr );
                }
                else
                {
                    success = -1;
                    break;
                }
            }
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE int NS(ManagedBuffer_remap)(
    SIXTRL_ARGPTR_DEC unsigned char* SIXTRL_RESTRICT begin,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_addr_diff_t) addr_diff_t;

    SIXTRL_STATIC_VAR addr_diff_t const ZERO_OFFSET = ( addr_diff_t )0;

    addr_diff_t base_addr_offset = ZERO_OFFSET;

    int success = NS(ManagedBuffer_get_addr_offset)(
        &base_addr_offset, begin, slot_size );

    if( ( success == 0 ) && ( base_addr_offset != ZERO_OFFSET ) )
    {
        addr_diff_t const offsets[] =
        {
            base_addr_offset, base_addr_offset, base_addr_offset,
            base_addr_offset, base_addr_offset
        };

        success  = NS(ManagedBuffer_remap_header)( begin, offsets, slot_size );

        success |= NS(ManagedBuffer_remap_section_slots)(
            begin, offsets, slot_size );

        success |= NS(ManagedBuffer_remap_section_objects)(
            begin, offsets, slot_size );

        success |= NS(ManagedBuffer_remap_section_dataptrs)(
            begin, offsets, slot_size );

        success |= NS(ManagedBuffer_remap_section_garbage)(
            begin, offsets, slot_size );
    }

    return success;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_IMPL_MANAGED_BUFFER_REMAP_H__ */

/*end: sixtracklib/common/impl/managed_buffer_remap.h */
