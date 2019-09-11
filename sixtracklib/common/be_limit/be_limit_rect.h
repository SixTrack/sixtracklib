#ifndef SIXTRL_COMMON_BE_LIMIT_RECT_C99_H__
#define SIXTRL_COMMON_BE_LIMIT_RECT_C99_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_limit/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(LimitRect)
{
    NS(particle_real_t) min_x SIXTRL_ALIGN( 8 );
    NS(particle_real_t) max_x SIXTRL_ALIGN( 8 );

    NS(particle_real_t) min_y SIXTRL_ALIGN( 8 );
    NS(particle_real_t) max_y SIXTRL_ALIGN( 8 );
}
NS(LimitRect);

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LimitRect_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN NS(buffer_size_t)
NS(LimitRect_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size );

SIXTRL_STATIC SIXTRL_FN SIXTRL_BE_ARGPTR_DEC NS(LimitRect)*
NS(LimitRect_preset)( SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_get_min_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit_rect );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_get_max_x)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit_rect );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_get_min_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit_rect );

SIXTRL_STATIC SIXTRL_FN NS(particle_real_t) NS(LimitRect_get_max_y)(
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const
        SIXTRL_RESTRICT limit_rect );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_set_x_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const x_limit );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_set_y_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const y_limit );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_set_min_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const min_x );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_set_max_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const max_x );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_set_min_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const min_y );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_set_max_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const max_y );

SIXTRL_STATIC SIXTRL_FN void NS(LimitRect_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect );

SIXTRL_STATIC SIXTRL_FN NS(arch_status_t) NS(LimitRect_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT source );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect) const*
NS(BufferIndex_get_const_limit_rect)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
NS(BufferIndex_get_limit_rect)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect) const*
NS(BeamElements_managed_buffer_get_const_limit_rect)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
NS(BeamElements_managed_buffer_get_limit_rect)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect) const*
NS(BeamElements_buffer_get_const_limit_rect)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
NS(BeamElements_buffer_get_limit_rect)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(LimitRect_get_required_num_dataptrs)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT  );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(buffer_size_t)
NS(LimitRect_get_required_num_slots)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC  const NS(LimitRect) *const SIXTRL_RESTRICT  );

SIXTRL_EXTERN SIXTRL_HOST_FN bool NS(LimitRect_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT req_dataptrs );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)* NS(LimitRect_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN
SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)* NS(LimitRect_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(particle_real_t) const min_x, NS(particle_real_t) const max_x,
    NS(particle_real_t) const min_y, NS(particle_real_t) const max_y );

SIXTRL_EXTERN SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
NS(LimitRect_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit );

#endif /* !defined( _GPUCODE )*/

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/*        Implementation of inline functions for NS(LimitRect)               */
/* ========================================================================= */

#if !defined( _GPUCODE )
    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( SIXTRL_NO_INCLUDES ) */
#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(buffer_size_t)
NS(LimitRect_get_required_num_dataptrs_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;
    ( void )limit;
    ( void )slot_size;

    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(LimitRect_get_required_num_slots_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT limit,
    NS(buffer_size_t) const slot_size )
{
    SIXTRL_ASSERT( buffer != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( slot_size > ( NS(buffer_size_t) )0u );

    ( void )buffer;

    return ( limit != SIXTRL_NULLPTR )
        ? NS(ManagedBuffer_get_slot_based_length)( sizeof( *limit ), slot_size )
        : ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* NS(LimitRect_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit )
{
    if( limit != SIXTRL_NULLPTR )
    {
        NS(LimitRect_set_min_x)( limit, SIXTRL_LIMIT_DEFAULT_MIN_X );
        NS(LimitRect_set_max_x)( limit, SIXTRL_LIMIT_DEFAULT_MAX_X );

        NS(LimitRect_set_min_y)( limit, SIXTRL_LIMIT_DEFAULT_MIN_Y );
        NS(LimitRect_set_max_y)( limit, SIXTRL_LIMIT_DEFAULT_MAX_Y );
    }

    return limit;
}

SIXTRL_INLINE  NS(particle_real_t) NS(LimitRect_get_min_x)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit_rect )
{
    SIXTRL_ASSERT( limit_rect != SIXTRL_NULLPTR );
    return limit_rect->min_x;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRect_get_max_x)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit_rect )
{
    SIXTRL_ASSERT( limit_rect != SIXTRL_NULLPTR );
    return limit_rect->max_x;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRect_get_min_y)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit_rect )
{
    SIXTRL_ASSERT( limit_rect != SIXTRL_NULLPTR );
    return limit_rect->min_y;
}

SIXTRL_INLINE NS(particle_real_t) NS(LimitRect_get_max_y)( SIXTRL_BE_ARGPTR_DEC
    const NS(LimitRect) *const SIXTRL_RESTRICT limit_rect )
{
    SIXTRL_ASSERT( limit_rect != SIXTRL_NULLPTR );
    return limit_rect->max_y;
}

SIXTRL_INLINE void NS(LimitRect_clear)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit )
{
    NS(LimitRect_preset)( limit );
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE void NS(LimitRect_set_x_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const x_limit )
{
    if( limit_rect != SIXTRL_NULLPTR )
    {
        limit_rect->min_x = -x_limit;
        limit_rect->max_x =  x_limit;
    }
}

SIXTRL_INLINE void NS(LimitRect_set_y_limit)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const y_limit )
{
    if( limit_rect != SIXTRL_NULLPTR )
    {
        limit_rect->min_y = -y_limit;
        limit_rect->max_y =  y_limit;
    }
}

SIXTRL_INLINE void NS(LimitRect_set_min_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const min_x )
{
    if( limit_rect != SIXTRL_NULLPTR ) limit_rect->min_x = min_x;
}

SIXTRL_INLINE void NS(LimitRect_set_max_x)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const max_x )
{
    if( limit_rect != SIXTRL_NULLPTR ) limit_rect->max_x = max_x;
}

SIXTRL_INLINE void NS(LimitRect_set_min_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const min_y )
{
    if( limit_rect != SIXTRL_NULLPTR ) limit_rect->min_y = min_y;
}

SIXTRL_INLINE void NS(LimitRect_set_max_y)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT limit_rect,
    NS(particle_real_t) const max_y )
{
    if( limit_rect != SIXTRL_NULLPTR ) limit_rect->max_y = max_y;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE NS(arch_status_t) NS(LimitRect_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(LimitRect)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(LimitRect) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source != SIXTRL_NULLPTR ) && ( destination != source ) )
    {
        if( destination != source )
        {
            destination->min_x = source->min_x;
            destination->max_x = source->max_x;

            destination->min_y = source->min_y;
            destination->max_y = source->max_y;
        }

        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect) const*
NS(BufferIndex_get_const_limit_rect)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj )
{
    typedef NS(LimitRect) beam_element_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t const* ptr_to_be_t;
    ptr_to_be_t ptr_to_be = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_LIMIT_RECT) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( beam_element_t ) ) )
    {
        ptr_to_be = ( ptr_to_be_t )( uintptr_t
            )NS(Object_get_begin_addr)( index_obj );
    }

    return ptr_to_be;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
NS(BufferIndex_get_limit_rect)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
        )NS(BufferIndex_get_const_limit_rect)( index_obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect) const*
NS(BeamElements_managed_buffer_get_const_limit_rect)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_const_limit_rect)(
        NS(ManagedBuffer_get_const_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
NS(BeamElements_managed_buffer_get_limit_rect)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_limit_rect)(
        NS(ManagedBuffer_get_object)( pbuffer, be_index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect) const*
NS(BeamElements_buffer_get_const_limit_rect)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_const_limit_rect)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(LimitRect)*
NS(BeamElements_buffer_get_limit_rect)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_limit_rect)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRL_COMMON_BE_LIMIT_RECT_C99_H__ */
/*end: sixtracklib/common/be_/be_.h */
