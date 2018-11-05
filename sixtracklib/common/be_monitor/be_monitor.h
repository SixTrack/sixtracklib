#ifndef SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef SIXTRL_INT64_T      NS(be_monitor_turn_t);
typedef SIXTRL_INT64_T      NS(be_monitor_flag_t);
typedef NS(buffer_addr_t)   NS(be_monitor_addr_t);
typedef SIXTRL_UINT64_T     NS(be_monitor_stride_t);

typedef struct NS(BeamMonitor)
{
    NS(be_monitor_turn_t)   num_stores        SIXTRL_ALIGN( 8 );
    NS(be_monitor_turn_t)   start             SIXTRL_ALIGN( 8 );
    NS(be_monitor_turn_t)   skip              SIXTRL_ALIGN( 8 );
    NS(be_monitor_addr_t)   io_address        SIXTRL_ALIGN( 8 );
    NS(be_monitor_stride_t) io_store_stride   SIXTRL_ALIGN( 8 );
    NS(be_monitor_flag_t)   rolling           SIXTRL_ALIGN( 8 );
    NS(be_monitor_flag_t)   cont_attributes   SIXTRL_ALIGN( 8 );
}
NS(BeamMonitor);

/* ------------------------------------------------------------------------- */
/* Helper functions: */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BeamMonitor_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(BeamMonitor_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC  const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor);

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear)(
    SIXTRL_BE_ARGPTR_DEC  NS(BeamMonitor)* SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT source );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT rhs );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* getter accessor functions: */

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_turn_t)
NS(BeamMonitor_get_num_stores)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_turn_t)
NS(BeamMonitor_get_start)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_turn_t)
NS(BeamMonitor_get_skip)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_is_rolling)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_are_attributes_continous)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_are_particles_continous)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_addr_t)
NS(BeamMonitor_get_io_address)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_stride_t)
NS(BeamMonitor_get_io_store_stride)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* setter accessor functions: */

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_num_stores)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const num_stores );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_start)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const start );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_skip)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const skip );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_is_rolling)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_rolling );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_are_attributes_continous)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const are_attributes_continous );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_io_address)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_addr_t) const io_address );

/*
SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_io_store_stride)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(be_monitor_stride_t) const stride );
*/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* Buffer management functions: */

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_monitor_turn_t) const num_stores, NS(be_monitor_turn_t) const start,
    NS(be_monitor_turn_t) const skip,  NS(be_monitor_addr_t) const io_address,
    NS(be_monitor_stride_t) const io_store_stride,
    bool const is_rolling, bool const are_attributes_continous );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #if !defined( _GPUCODE )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( _GPUCODE ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* Helper functions: */

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_get_num_dataptrs)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    return ( NS(buffer_size_t) )0u;
}

SIXTRL_INLINE NS(buffer_size_t) NS(BeamMonitor_get_num_slots)(
    SIXTRL_BE_ARGPTR_DEC  const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(BeamMonitor)   beam_element_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    ( void )monitor;

    buf_size_t extent = NS(ManagedBuffer_get_slot_based_length)(
        sizeof( beam_element_t ), slot_size );

    SIXTRL_ASSERT( ( slot_size == ZERO ) || ( ( extent % slot_size ) == ZERO ) );
    return ( slot_size > ZERO ) ? ( extent / slot_size ) : ( ZERO );
}

SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_preset)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor)
{
    if( monitor != SIXTRL_NULLPTR )
    {
        NS(BeamMonitor_set_num_stores)( monitor, 0u );
        NS(BeamMonitor_set_start)( monitor, 0u );
        NS(BeamMonitor_set_skip)(  monitor, 1u );
        NS(BeamMonitor_set_is_rolling)( monitor, false );
        NS(BeamMonitor_set_are_attributes_continous)( monitor, true );

        NS(BeamMonitor_clear)( monitor );
    }

    return monitor;
}

SIXTRL_INLINE void NS(BeamMonitor_clear)(
    SIXTRL_BE_ARGPTR_DEC  NS(BeamMonitor)* SIXTRL_RESTRICT monitor )
{
    NS(BeamMonitor_set_io_address)( monitor, ( NS(buffer_addr_t) )0 );
    //NS(BeamMonitor_set_io_store_stride)( monitor, ( NS(buffer_size_t) )0 );
    if( monitor != SIXTRL_NULLPTR ) monitor->io_store_stride = 0u;

    return;
}

SIXTRL_INLINE int NS(BeamMonitor_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT source )
{
    int success = -1;

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source != SIXTRL_NULLPTR ) )
    {
        NS(BeamMonitor_set_num_stores)(destination,
             NS(BeamMonitor_get_num_stores)( source ) );

        NS(BeamMonitor_set_start)(destination,
             NS(BeamMonitor_get_start)( source ) );

        NS(BeamMonitor_set_skip)( destination,
             NS(BeamMonitor_get_skip)(  source ) );

        NS(BeamMonitor_set_is_rolling)( destination,
             NS(BeamMonitor_is_rolling)( source ) );

        NS(BeamMonitor_set_are_attributes_continous)( destination,
             NS(BeamMonitor_are_attributes_continous)( source ) );

        NS(BeamMonitor_set_io_address)( destination,
            NS(BeamMonitor_get_io_address)( source ) );

        destination->io_store_stride =
            NS(BeamMonitor_get_io_store_stride)( source );

        success = 0;
    }

    return success;

}

SIXTRL_INLINE int NS(BeamMonitor_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT rhs )
{
    int compare_value = -1;

    if( ( lhs != SIXTRL_NULLPTR ) && ( rhs != SIXTRL_NULLPTR ) )
    {
        if( NS(BeamMonitor_get_num_stores)( lhs ) ==
            NS(BeamMonitor_get_num_stores)( rhs ) )
        {
            compare_value = 0;
        }
        else if( NS(BeamMonitor_get_num_stores)( lhs ) >
                 NS(BeamMonitor_get_num_stores)( rhs ) )
        {
            compare_value = +1;
        }
        else if( NS(BeamMonitor_get_num_stores)( lhs ) <
                 NS(BeamMonitor_get_num_stores)( rhs ) )
        {
            compare_value = -1;
        }

        if( compare_value == 0 )
        {
            if( NS(BeamMonitor_get_start)( lhs ) >
                NS(BeamMonitor_get_start)( rhs ) )
            {
                compare_value = +1;
            }
            else if( NS(BeamMonitor_get_start)( lhs ) <
                     NS(BeamMonitor_get_start)( rhs ) )
            {
                compare_value = -1;
            }
        }

        if( compare_value == 0 )
        {
            if( NS(BeamMonitor_get_skip)( lhs ) >
                NS(BeamMonitor_get_skip)( rhs ) )
            {
                compare_value = +1;
            }
            else if( NS(BeamMonitor_get_skip)( lhs ) <
                     NS(BeamMonitor_get_skip)( rhs ) )
            {
                compare_value = -1;
            }
        }

        if( compare_value == 0 )
        {
            if( lhs->rolling > rhs->rolling )
            {
                compare_value = +1;
            }
            else if( lhs->rolling < rhs->rolling )
            {
                compare_value = -1;
            }
        }

        if( compare_value == 0 )
        {
            if( NS(BeamMonitor_get_io_address)( lhs ) >
                NS(BeamMonitor_get_io_address)( rhs ) )
            {
                compare_value = +1;
            }
            else if( NS(BeamMonitor_get_io_address)( lhs ) <
                     NS(BeamMonitor_get_io_address)( rhs ) )
            {
                compare_value = -1;
            }
        }

        if( compare_value == 0 )
        {
            if( NS(BeamMonitor_get_io_store_stride)( lhs ) >
                NS(BeamMonitor_get_io_store_stride)( rhs ) )
            {
                compare_value = +1;
            }
            else if( NS(BeamMonitor_get_io_store_stride)( lhs ) <
                     NS(BeamMonitor_get_io_store_stride)( rhs ) )
            {
                compare_value = -1;
            }
        }

        if( compare_value == 0 )
        {
            if( lhs->cont_attributes > rhs->cont_attributes )
            {
                compare_value = +1;
            }
            else if( lhs->cont_attributes < rhs->cont_attributes )
            {
                compare_value = -1;
            }
        }
    }
    else if( lhs != SIXTRL_NULLPTR )
    {
        compare_value = +1;
    }

    return compare_value;
}

SIXTRL_INLINE int NS(BeamMonitor_compare_values_with_treshold)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT rhs,
    SIXTRL_REAL_T const treshold )
{
    ( void )treshold;
    return NS(BeamMonitor_compare_values)( lhs, rhs );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* getter accessor functions: */

SIXTRL_INLINE NS(be_monitor_turn_t)
NS(BeamMonitor_get_num_stores)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->num_stores;
}

SIXTRL_INLINE NS(be_monitor_turn_t)
NS(BeamMonitor_get_start)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->start;
}

SIXTRL_INLINE NS(be_monitor_turn_t)
NS(BeamMonitor_get_skip)( SIXTRL_BE_ARGPTR_DEC const
    NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->skip;
}

SIXTRL_INLINE bool NS(BeamMonitor_is_rolling)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return ( monitor->rolling == 1 );
}

SIXTRL_INLINE bool NS(BeamMonitor_are_attributes_continous)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return ( monitor->cont_attributes == 1 );
}

SIXTRL_INLINE bool NS(BeamMonitor_are_particles_continous)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    return !NS(BeamMonitor_are_attributes_continous)( monitor );
}

SIXTRL_INLINE NS(be_monitor_addr_t)
NS(BeamMonitor_get_io_address)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->io_address;
}

SIXTRL_INLINE NS(be_monitor_stride_t)
NS(BeamMonitor_get_io_store_stride)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->io_store_stride;
}


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* setter accessor functions: */

SIXTRL_INLINE void NS(BeamMonitor_set_num_stores)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const num_stores )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->num_stores = num_stores;
    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_start)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const start )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->start = start;
    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_skip)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const skip )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->skip = skip;
    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_is_rolling)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_rolling )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->rolling = ( is_rolling ) ? 1 : 0;
    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_are_attributes_continous)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const are_attributes_continous )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->cont_attributes =
            ( are_attributes_continous ) ? 1 : 0;

    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_io_address)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_addr_t) const io_address )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->io_address = io_address;
    return;
}

/*
SIXTRL_INLINE void NS(BeamMonitor_set_io_store_stride)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_stride_t) const io_store_stride )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->io_store_stride = io_store_stride;
    return;
}
*/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* Buffer management functions: */

#if !defined( _GPUCODE )

SIXTRL_INLINE bool NS(BeamMonitor_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const num_dataptrs =
        NS(BeamMonitor_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes  = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts = SIXTRL_NULLPTR;

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(BeamMonitor) ),
        num_dataptrs, sizes, counts, requ_objects, requ_slots, requ_dataptrs );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(BeamMonitor)                         elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC   elem_t*     ptr_elem_t;
    typedef NS(be_monitor_turn_t)                   nturn_t;
    typedef NS(be_monitor_flag_t)                   flag_t;
    typedef NS(be_monitor_addr_t)                   addr_t;

    buf_size_t const num_dataptrs =
        NS(BeamMonitor_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.num_stores        = ( nturn_t )0u;
    temp_obj.start             = ( nturn_t )0u;
    temp_obj.skip              = ( nturn_t )1u;
    temp_obj.rolling           = ( flag_t )0u;
    temp_obj.io_address        = ( addr_t )0u;
    temp_obj.io_store_stride   = ( NS(be_monitor_stride_t) )0u;
    temp_obj.cont_attributes   = ( flag_t )1;

    return ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( temp_obj ),
        NS(OBJECT_TYPE_BEAM_MONITOR), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_monitor_turn_t) const num_stores, NS(be_monitor_turn_t) const start,
    NS(be_monitor_turn_t) const skip,  NS(be_monitor_addr_t) const io_address,
    NS(be_monitor_stride_t) const io_store_stride,
    bool const is_rolling, bool const are_attributes_continous )
{
    typedef NS(buffer_size_t)                       buf_size_t;
    typedef NS(BeamMonitor)                         elem_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC   elem_t*     ptr_elem_t;

    buf_size_t const num_dataptrs =
        NS(BeamMonitor_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    SIXTRL_ASSERT( io_store_stride == ( NS(be_monitor_stride_t) )0u );

    elem_t temp_obj;
    temp_obj.num_stores        = num_stores;
    temp_obj.start             = start;
    temp_obj.skip              = skip;
    temp_obj.rolling           = ( is_rolling ) ? 1 : 0;
    temp_obj.io_address        = io_address;
    temp_obj.io_store_stride   = io_store_stride;
    temp_obj.cont_attributes   = ( are_attributes_continous ) ? 1 : 0;

    return ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( temp_obj ),
        NS(OBJECT_TYPE_BEAM_MONITOR), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    return NS(BeamMonitor_add)( buffer,
        NS(BeamMonitor_get_num_stores)( monitor ),
        NS(BeamMonitor_get_start)( monitor ),
        NS(BeamMonitor_get_skip)( monitor ),
        NS(BeamMonitor_get_io_address)( monitor ),
        NS(BeamMonitor_get_io_store_stride)( monitor ),
        NS(BeamMonitor_is_rolling)( monitor ),
        NS(BeamMonitor_are_attributes_continous)( monitor ) );
}

#endif /* !defined( _GPUCODE ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__ */

/* end: sixtracklib/common/be_monitor/be_monitor.h */
