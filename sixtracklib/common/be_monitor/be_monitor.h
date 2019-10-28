#ifndef SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__
#define SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/internal/buffer_main_defines.h"
    #include "sixtracklib/common/internal/buffer_object_defines.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #include "sixtracklib/common/buffer/buffer_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef SIXTRL_INT64_T      NS(be_monitor_turn_t);
typedef SIXTRL_INT64_T      NS(be_monitor_flag_t);
typedef SIXTRL_INT64_T      NS(be_monitor_index_t);
typedef NS(buffer_addr_t)   NS(be_monitor_addr_t);

typedef struct NS(BeamMonitor)
{
    NS(be_monitor_turn_t)   num_stores        SIXTRL_ALIGN( 8 );
    NS(be_monitor_turn_t)   start             SIXTRL_ALIGN( 8 );
    NS(be_monitor_turn_t)   skip              SIXTRL_ALIGN( 8 );
    NS(be_monitor_addr_t)   out_address       SIXTRL_ALIGN( 8 );
    NS(be_monitor_index_t)  max_particle_id   SIXTRL_ALIGN( 8 );
    NS(be_monitor_index_t)  min_particle_id   SIXTRL_ALIGN( 8 );
    NS(be_monitor_flag_t)   is_rolling        SIXTRL_ALIGN( 8 );
    NS(be_monitor_flag_t)   is_turn_ordered   SIXTRL_ALIGN( 8 );
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

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all_line_obj)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* be_end );

SIXTRL_FN SIXTRL_STATIC NS(arch_status_t) NS(BeamMonitor_copy)(
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

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_is_turn_ordered)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_is_particle_ordered)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_addr_t)
NS(BeamMonitor_get_out_address)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_index_t)
NS(BeamMonitor_get_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(be_monitor_index_t)
NS(BeamMonitor_get_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_stored_num_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* accessor functions for retrieving already dumped data */

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_has_turn_stored)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const turn_id,
    NS(be_monitor_turn_t) const max_num_turns );

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

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_is_turn_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_turn_ordered );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_is_particle_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_particle_ordered );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_out_address)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_addr_t) const out_address );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const min_particle_id );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_set_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const min_particle_id );

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* Buffer management functions: */

SIXTRL_FN SIXTRL_STATIC NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* index_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* index_end,
    NS(buffer_size_t) const start_index,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors );

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const*
NS(BufferIndex_get_const_beam_monitor)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BufferIndex_get_beam_monitor)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const*
NS(BeamElements_managed_buffer_get_const_beam_monitor)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamElements_managed_buffer_get_beam_monitor)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const*
NS(BeamElements_buffer_get_const_beam_monitor)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_STATIC SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamElements_buffer_get_beam_monitor)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index );

SIXTRL_FN SIXTRL_STATIC bool
NS(BeamMonitor_are_present_in_buffer)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT belements_buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT belements_buffer );

SIXTRL_FN SIXTRL_STATIC NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors );


SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer );

SIXTRL_FN SIXTRL_STATIC bool NS(BeamMonitor_can_be_added)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_objects,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_slots,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_new)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_monitor_turn_t) const num_stores, NS(be_monitor_turn_t) const start,
    NS(be_monitor_turn_t) const skip,  NS(be_monitor_addr_t) const out_address,
    NS(be_monitor_index_t) const min_particle_id,
    NS(be_monitor_index_t) const max_particle_id,
    bool const is_rolling, bool const is_turn_ordered );

SIXTRL_FN SIXTRL_STATIC SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamMonitor_add_copy)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_insert_end_of_turn_monitors)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* SIXTRL_RESTRICT prev_node );

SIXTRL_EXTERN SIXTRL_HOST_FN NS(arch_status_t)
NS(BeamMonitor_insert_end_of_turn_monitors_at_pos)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(be_monitor_turn_t) const turn_by_turn_start,
    NS(be_monitor_turn_t) const num_turn_by_turn_turns,
    NS(be_monitor_turn_t) const target_num_turns,
    NS(be_monitor_turn_t) const skip_turns,
    NS(buffer_size_t) const insert_at_index );

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT
        blements_buffer, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT
        beam_elements_buffer, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all_on_managed_buffer_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT
        beam_elements_buffer,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags );

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
        NS(BeamMonitor_set_is_turn_ordered)( monitor, true );
        NS(BeamMonitor_clear)( monitor );
    }

    return monitor;
}

SIXTRL_INLINE void NS(BeamMonitor_clear)(
    SIXTRL_BE_ARGPTR_DEC  NS(BeamMonitor)* SIXTRL_RESTRICT monitor )
{
    NS(BeamMonitor_set_out_address)( monitor, ( NS(buffer_addr_t) )0 );
    NS(BeamMonitor_set_min_particle_id)( monitor, ( NS(be_monitor_index_t) )0 );
    NS(BeamMonitor_set_max_particle_id)( monitor, ( NS(be_monitor_index_t) )0 );

    return;
}

SIXTRL_INLINE void NS(BeamMonitor_clear_all_line_obj)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* be_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* be_end )
{
    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( ( uintptr_t )be_it ) <= ( ( uintptr_t )be_end ) ) );

    for( ; be_it != be_end ; ++be_it )
    {
        if( NS(Object_get_type_id)( be_it ) == NS(OBJECT_TYPE_BEAM_MONITOR) )
        {
            SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* beam_monitor =
                ( SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* )( uintptr_t
                    )NS(Object_get_begin_addr)( be_it );

            SIXTRL_ASSERT( NS(Object_get_size)( be_it ) >=
                           sizeof( NS(BeamMonitor ) ) );

            SIXTRL_ASSERT( beam_monitor != SIXTRL_NULLPTR );

            NS(BeamMonitor_clear)( beam_monitor );
        }
    }

    return;
}

SIXTRL_INLINE NS(arch_status_t) NS(BeamMonitor_copy)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT destination,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT source )
{
    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

    if( ( destination != SIXTRL_NULLPTR ) &&
        ( source != SIXTRL_NULLPTR ) )
    {
        NS(BeamMonitor_set_num_stores)(destination,
             NS(BeamMonitor_get_num_stores)( source ) );

        NS(BeamMonitor_set_start)(destination,
             NS(BeamMonitor_get_start)( source ) );

        NS(BeamMonitor_set_skip)( destination,
             NS(BeamMonitor_get_skip)(  source ) );

        NS(BeamMonitor_set_out_address)( destination,
            NS(BeamMonitor_get_out_address)( source ) );

        NS(BeamMonitor_set_is_rolling)( destination,
             NS(BeamMonitor_is_rolling)( source ) );

        NS(BeamMonitor_set_is_turn_ordered)( destination,
            NS(BeamMonitor_is_turn_ordered)( source ) );

        NS(BeamMonitor_set_min_particle_id)( destination,
            NS(BeamMonitor_get_min_particle_id)( source ) );

        NS(BeamMonitor_set_max_particle_id)( destination,
            NS(BeamMonitor_get_max_particle_id)( source ) );

        status = SIXTRL_ARCH_STATUS_SUCCESS;
    }

    return status;

}

SIXTRL_INLINE int NS(BeamMonitor_compare_values)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT lhs,
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT rhs )
{
    typedef NS(be_monitor_turn_t)   nturn_t;
    typedef NS(be_monitor_index_t)  index_t;
    typedef NS(be_monitor_addr_t)   addr_t;

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
            nturn_t const lhs_value = NS(BeamMonitor_get_start)( lhs );
            nturn_t const rhs_value = NS(BeamMonitor_get_start)( rhs );

            if( lhs_value != rhs_value )
            {
                if( lhs_value > rhs_value )
                {
                    compare_value = +1;
                }
                else if( lhs_value < rhs_value )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            nturn_t const lhs_value = NS(BeamMonitor_get_skip)( lhs );
            nturn_t const rhs_value = NS(BeamMonitor_get_skip)( rhs );

            if( lhs_value != rhs_value )
            {
                if( lhs_value > rhs_value )
                {
                    compare_value = +1;
                }
                else if( lhs_value < rhs_value )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            addr_t const lhs_addr = NS(BeamMonitor_get_out_address)( lhs );
            addr_t const rhs_addr = NS(BeamMonitor_get_out_address)( rhs );

            if( lhs_addr != rhs_addr )
            {
                if( lhs_addr > rhs_addr )
                {
                    compare_value = +1;
                }
                else if( lhs_addr < rhs_addr )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            index_t const lhs_value = NS(BeamMonitor_get_min_particle_id)( lhs );
            index_t const rhs_value = NS(BeamMonitor_get_min_particle_id)( rhs );

            if( lhs_value != rhs_value )
            {
                if( lhs_value > rhs_value )
                {
                    compare_value = +1;
                }
                else if( lhs_value < rhs_value )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            index_t const lhs_value = NS(BeamMonitor_get_max_particle_id)( lhs );
            index_t const rhs_value = NS(BeamMonitor_get_max_particle_id)( rhs );

            if( lhs_value != rhs_value )
            {
                if( lhs_value > rhs_value )
                {
                    compare_value = +1;
                }
                else if( lhs_value < rhs_value )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            if( lhs->is_rolling != rhs->is_rolling )
            {
                if( lhs->is_rolling > rhs->is_rolling )
                {
                    compare_value = +1;
                }
                else if( lhs->is_rolling < rhs->is_rolling )
                {
                    compare_value = -1;
                }
            }
        }

        if( compare_value == 0 )
        {
            if( lhs->is_turn_ordered != rhs->is_turn_ordered )
            {
                if( lhs->is_turn_ordered > rhs->is_turn_ordered )
                {
                    compare_value = +1;
                }
                else if( lhs->is_turn_ordered < rhs->is_turn_ordered )
                {
                    compare_value = -1;
                }
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
    return ( monitor->is_rolling == 1 );
}

SIXTRL_INLINE bool NS(BeamMonitor_is_turn_ordered)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return ( monitor->is_turn_ordered == 1 );
}

SIXTRL_INLINE bool NS(BeamMonitor_is_particle_ordered)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    return !NS(BeamMonitor_is_turn_ordered)( monitor );
}

SIXTRL_INLINE NS(be_monitor_addr_t)
NS(BeamMonitor_get_out_address)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->out_address;
}

SIXTRL_INLINE NS(be_monitor_index_t)
NS(BeamMonitor_get_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    return monitor->min_particle_id;
}

SIXTRL_INLINE NS(be_monitor_index_t)
NS(BeamMonitor_get_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    SIXTRL_ASSERT( monitor != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( monitor->max_particle_id >= monitor->min_particle_id );
    return monitor->max_particle_id;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_get_stored_num_particles)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor )
{
    typedef NS(buffer_size_t)       size_t;
    typedef NS(be_monitor_index_t)  index_t;

    size_t required_num_particles = ( size_t )0u;

    index_t const min_particle_id =
        NS(BeamMonitor_get_min_particle_id)( monitor );

    index_t const max_particle_id =
        NS(BeamMonitor_get_max_particle_id)( monitor );

    if( ( min_particle_id >= ( index_t )0u ) &&
        ( max_particle_id >= min_particle_id ) )
    {
        required_num_particles = ( size_t )(
            max_particle_id - min_particle_id + ( size_t )1u );

        required_num_particles *= NS(BeamMonitor_get_num_stores)( monitor );
    }

    return required_num_particles;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* accessor functions for retrieving already dumped data */

SIXTRL_INLINE bool NS(BeamMonitor_has_turn_stored)(
    SIXTRL_BE_ARGPTR_DEC const NS(BeamMonitor) *const SIXTRL_RESTRICT monitor,
    NS(be_monitor_turn_t) const turn_id,
    NS(be_monitor_turn_t) const max_num_turns )
{
    bool is_stored = false;

    typedef NS(be_monitor_turn_t) nturn_t;

    nturn_t const num_stores = NS(BeamMonitor_get_num_stores)( monitor );
    nturn_t const start_turn = NS(BeamMonitor_get_start)( monitor );
    nturn_t const skip_turns = NS(BeamMonitor_get_skip)( monitor );

    if( ( monitor != SIXTRL_NULLPTR ) && ( start_turn >= ( nturn_t )0u ) &&
        ( start_turn <= turn_id ) && ( num_stores > ( nturn_t )0u ) &&
        ( skip_turns > ( nturn_t )0u ) &&
        ( ( max_num_turns >= turn_id ) || ( max_num_turns < ( nturn_t )0u ) ) )
    {
        if( turn_id >= start_turn )
        {
            nturn_t turns_since_start = turn_id - start_turn;

            if( ( turns_since_start % skip_turns ) == ( nturn_t )0u )
            {
                nturn_t store_idx = turns_since_start / skip_turns;

                if( !NS(BeamMonitor_is_rolling)( monitor ) )
                {
                    if( store_idx < num_stores ) is_stored = true;
                }
                else if( max_num_turns > start_turn )
                {
                    nturn_t const max_turn = max_num_turns - start_turn;
                    nturn_t max_num_stores = max_turn / skip_turns;

                    if( ( max_turn % skip_turns ) != ( nturn_t )0u )
                    {
                        ++max_num_stores;
                    }

                    if( max_num_stores <= ( store_idx + num_stores ) )
                    {
                        is_stored = true;
                    }
                }
            }
        }
    }

    return is_stored;
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
    if( monitor != SIXTRL_NULLPTR )
        monitor->is_rolling = ( is_rolling ) ? 1 : 0;
    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_is_turn_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_turn_ordered )
{
    if( monitor != SIXTRL_NULLPTR )
        monitor->is_turn_ordered = ( is_turn_ordered ) ? 1 : 0;

    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_is_particle_ordered)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    bool const is_particle_ordered )
{
    if( monitor != SIXTRL_NULLPTR )
        monitor->is_turn_ordered = ( is_particle_ordered ) ? 0 : 1;

    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_out_address)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_addr_t) const out_address )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->out_address = out_address;
    return;
}

SIXTRL_INLINE void NS(BeamMonitor_set_min_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const min_particle_id )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->min_particle_id = min_particle_id;
}

SIXTRL_INLINE void NS(BeamMonitor_set_max_particle_id)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT monitor,
    NS(be_monitor_index_t) const max_particle_id )
{
    if( monitor != SIXTRL_NULLPTR ) monitor->max_particle_id = max_particle_id;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* Buffer management functions: */


SIXTRL_INLINE NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors,
    NS(buffer_size_t) const slot_s )
{
    typedef NS(buffer_size_t) buf_size_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( buffer != SIXTRL_NULLPTR ) && ( slot_s > ZERO ) &&
        ( !NS(ManagedBuffer_needs_remapping)( buffer, slot_s ) ) &&
        ( NS(ManagedBuffer_get_num_objects)( buffer, slot_s ) > ZERO ) )
    {
        status = NS(BeamMonitor_get_beam_monitor_indices_from_index_range)(
            NS(ManagedBuffer_get_const_objects_index_begin)( buffer, slot_s ),
            NS(ManagedBuffer_get_const_objects_index_end)( buffer, slot_s ),
            ZERO, max_num_of_indices, indices_begin, ptr_num_be_monitors );
    }

    return status;
}

SIXTRL_INLINE NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_index_range)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* index_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* index_end,
    NS(buffer_size_t) const start_index,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors )
{
    typedef NS(buffer_size_t)    buf_size_t;
    typedef NS(buffer_addr_t)    address_t;
    typedef NS(object_type_id_t) obj_type_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const* ptr_be_monitor_t;

    NS(arch_status_t) status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    SIXTRL_STATIC_VAR buf_size_t const MON_SIZE = sizeof( NS(BeamMonitor) );
    SIXTRL_STATIC_VAR address_t  const ZERO_ADDR = ( address_t )0u;
    buf_size_t num_be_monitors = ZERO;

    if( ( index_it != SIXTRL_NULLPTR ) && ( index_end != SIXTRL_NULLPTR ) &&
        ( ( ( uintptr_t )index_end ) >= ( ( uintptr_t )index_it ) ) &&
        ( max_num_of_indices > ZERO ) &&
        ( indices_begin != SIXTRL_NULLPTR ) )
    {
        buf_size_t next_index = start_index;

        status = SIXTRL_ARCH_STATUS_SUCCESS;

        while( ( num_be_monitors < max_num_of_indices ) &&
               ( index_it != index_end ) )
        {
            obj_type_t const type_id = NS(Object_get_type_id)( index_it );
            address_t const addr = NS(Object_get_begin_addr)( index_it );

            if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR ) ) &&
                ( addr > ZERO_ADDR ) &&
                ( NS(Object_get_size)( index_it ) >= MON_SIZE ) )
            {
                ptr_be_monitor_t mon = ( ptr_be_monitor_t )( uintptr_t )addr;

                if( mon != SIXTRL_NULLPTR )
                {
                    indices_begin[ num_be_monitors++ ] = next_index;
                }
            }
            else if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
                break;
            }

            ++next_index;
            ++index_it;
        }
    }

    if( ( status == SIXTRL_ARCH_STATUS_SUCCESS ) &&
        ( num_be_monitors == max_num_of_indices ) &&
        ( index_it != index_end ) )
    {
        status = SIXTRL_ARCH_STATUS_GENERAL_FAILURE;
    }

    if(  ptr_num_be_monitors != SIXTRL_NULLPTR )
    {
        *ptr_num_be_monitors = num_be_monitors;
    }

    return status;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const*
NS(BufferIndex_get_const_beam_monitor)(
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const index_obj )
{
    typedef NS(BeamMonitor) beam_element_t;
    typedef SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t const* ptr_to_be_t;
    ptr_to_be_t ptr_to_be = SIXTRL_NULLPTR;

    if( ( index_obj != SIXTRL_NULLPTR ) &&
        ( NS(Object_get_type_id)( index_obj ) == NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
        ( NS(Object_get_size)( index_obj ) >= sizeof( beam_element_t ) ) )
    {
        ptr_to_be = ( ptr_to_be_t )( uintptr_t
            )NS(Object_get_begin_addr)( index_obj );
    }

    return ptr_to_be;
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BufferIndex_get_beam_monitor)( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* index_obj )
{
    return ( SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
        )NS(BufferIndex_get_const_beam_monitor)( index_obj );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const*
NS(BeamElements_managed_buffer_get_const_beam_monitor)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_const_beam_monitor)(
        NS(ManagedBuffer_get_const_object)( pbuffer, be_index, slot_size ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamElements_managed_buffer_get_beam_monitor)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    NS(buffer_size_t) const be_index, NS(buffer_size_t) const slot_size )
{
    return NS(BufferIndex_get_beam_monitor)(
        NS(ManagedBuffer_get_object)( pbuffer, be_index, slot_size ) );
}

#if !defined( _GPUCODE )

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor) const*
NS(BeamElements_buffer_get_const_beam_monitor)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_const_beam_monitor)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)*
NS(BeamElements_buffer_get_beam_monitor)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const be_index )
{
    typedef SIXTRL_BUFFER_DATAPTR_DEC unsigned char* ptr_raw_t;
    return NS(BeamElements_managed_buffer_get_beam_monitor)(
        ( ptr_raw_t )( uintptr_t )NS(Buffer_get_data_begin_addr)( buffer ),
        be_index, NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE bool NS(BeamMonitor_are_present_in_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT
        beam_elements_buffer )
{
    bool beam_monitors_are_present = false;

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj_it =
        NS(Buffer_get_const_objects_begin)( beam_elements_buffer );

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* obj_end =
        NS(Buffer_get_const_objects_end)( beam_elements_buffer );

    for( ; obj_it != obj_end ; ++obj_it )
    {
        if( ( NS(Object_get_type_id)( obj_it ) ==
                NS(OBJECT_TYPE_BEAM_MONITOR ) ) &&
            ( ( uintptr_t )NS(Object_get_begin_addr)( obj_it ) !=
              ( uintptr_t )0u ) )
        {
            typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor) const* ptr_monitor_t;
            typedef NS(be_monitor_turn_t) nturn_t;

            ptr_monitor_t monitor = ( ptr_monitor_t
                )( uintptr_t )NS(Object_get_begin_addr)( obj_it );

            if( ( monitor != SIXTRL_NULLPTR ) &&
                ( NS(BeamMonitor_get_num_stores)( monitor ) > ( nturn_t )0u ) )
            {
                beam_monitors_are_present = true;
                break;
            }
        }
    }

    return beam_monitors_are_present;
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer )
{
    return NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
        NS(Buffer_get_const_data_begin)( beam_elements_buffer ),
        NS(Buffer_get_slot_size)( beam_elements_buffer ) );
}

SIXTRL_INLINE NS(arch_status_t)
NS(BeamMonitor_get_beam_monitor_indices_from_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(buffer_size_t) const max_num_of_indices,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT indices_begin,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_be_monitors )
{
    return NS(BeamMonitor_get_beam_monitor_indices_from_managed_buffer)(
        NS(Buffer_get_const_data_begin)( buffer ),
        max_num_of_indices, indices_begin, ptr_num_be_monitors,
        NS(Buffer_get_slot_size)( buffer ) );
}

SIXTRL_INLINE void NS(BeamMonitor_clear_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer )
{
    NS(BeamMonitor_clear_all_on_managed_buffer)(
        NS(Buffer_get_data_begin)( beam_elements_buffer),
        NS(Buffer_get_slot_size)( beam_elements_buffer ) );

    return;
}


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
    temp_obj.out_address       = ( addr_t )0u;
    temp_obj.min_particle_id   = ( NS(be_monitor_index_t) )0u;
    temp_obj.max_particle_id   = ( NS(be_monitor_index_t) )0u;
    temp_obj.is_rolling        = ( flag_t )0;
    temp_obj.is_turn_ordered   = ( flag_t )1;

    return ( ptr_elem_t )( uintptr_t )NS(Object_get_begin_addr)(
        NS(Buffer_add_object)( buffer, &temp_obj, sizeof( temp_obj ),
        NS(OBJECT_TYPE_BEAM_MONITOR), num_dataptrs, offsets, sizes, counts ) );
}

SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC NS(BeamMonitor)* NS(BeamMonitor_add)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(be_monitor_turn_t) const num_stores, NS(be_monitor_turn_t) const start,
    NS(be_monitor_turn_t) const skip,  NS(be_monitor_addr_t) const out_address,
    NS(be_monitor_index_t) const min_particle_id,
    NS(be_monitor_index_t) const max_particle_id,
    bool const is_rolling, bool const is_turn_ordered )
{
    typedef NS(BeamMonitor) elem_t;
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC elem_t* ptr_elem_t;

    buf_size_t const num_dataptrs =
        NS(BeamMonitor_get_num_dataptrs)( SIXTRL_NULLPTR );

    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* offsets = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* sizes   = SIXTRL_NULLPTR;
    SIXTRL_BUFFER_ARGPTR_DEC buf_size_t const* counts  = SIXTRL_NULLPTR;

    elem_t temp_obj;
    temp_obj.num_stores        = num_stores;
    temp_obj.start             = start;
    temp_obj.skip              = skip;
    temp_obj.out_address       = out_address;
    temp_obj.min_particle_id   = min_particle_id;
    temp_obj.max_particle_id   = max_particle_id;
    temp_obj.is_rolling        = ( is_rolling      ) ? 1 : 0;
    temp_obj.is_turn_ordered   = ( is_turn_ordered ) ? 1 : 0;

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
        NS(BeamMonitor_get_out_address)( monitor ),
        NS(BeamMonitor_get_min_particle_id)( monitor ),
        NS(BeamMonitor_get_max_particle_id)( monitor ),
        NS(BeamMonitor_is_rolling)( monitor ),
        NS(BeamMonitor_is_turn_ordered)( monitor ) );
}

#endif /* !defined( _GPUCODE ) */

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT beam_elements, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    buf_size_t num_beam_monitors = ZERO;

    if( ( beam_elements != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( NS(ManagedBuffer_get_num_objects)(
            beam_elements, slot_size ) ) > ZERO )
    {
        ptr_obj_t be_it  = NS(ManagedBuffer_get_const_objects_index_begin)(
            beam_elements, slot_size );

        ptr_obj_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            beam_elements, slot_size );

        for( ; be_it != be_end ; ++be_it )
        {
            NS(object_type_id_t) const type_id =
                NS(Object_get_type_id)( be_it );

            if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
                ( NS(Object_get_const_begin_ptr)( be_it ) !=
                    SIXTRL_NULLPTR ) &&
                ( NS(Object_get_size)( be_it ) >= sizeof( NS(BeamMonitor) ) ) )
            {
                ++num_beam_monitors;
            }
        }
    }

    return num_beam_monitors;
}

SIXTRL_INLINE void NS(BeamMonitor_clear_all_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* ptr_beam_monitor_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    if( ( beam_elements != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( NS(ManagedBuffer_get_num_objects)(
            beam_elements, slot_size ) ) > ZERO )
    {
        ptr_obj_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
            beam_elements, slot_size );

        ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            beam_elements, slot_size );

        for( ; be_it != be_end ; ++be_it )
        {
            NS(object_type_id_t) const type_id =
                NS(Object_get_type_id)( be_it );

            if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t
                    )NS(Object_get_begin_ptr)( be_it );

                if( monitor != SIXTRL_NULLPTR )
                {
                    NS(BeamMonitor_clear)( monitor );
                }
            }
        }
    }

    return;
}

SIXTRL_INLINE void NS(BeamMonitor_clear_all_on_managed_buffer_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const slot_size,
    SIXTRL_ARGPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_status_flags )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* ptr_beam_monitor_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    NS(arch_debugging_t) flags = SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

    NS(arch_debugging_t) const BELEM_BUFFER_NULL_FLAG           = flags;
    NS(arch_debugging_t) const SLOT_SIZE_ILLEGAL_FLAG           = flags <<  1u;
    NS(arch_debugging_t) const BELEM_BUFFER_REQUIRES_REMAP_FLAG = flags <<  2u;
    NS(arch_debugging_t) const BELEM_BUFFER_EMPTY_FLAG          = flags <<  3u;

    flags = ( NS(arch_debugging_t) )0u;

    if( ( beam_elements != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( NS(ManagedBuffer_get_num_objects)(
            beam_elements, slot_size ) ) > ZERO )
    {
        ptr_obj_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
            beam_elements, slot_size );

        ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            beam_elements, slot_size );

        for( ; be_it != be_end ; ++be_it )
        {
            NS(object_type_id_t) const type_id =
                NS(Object_get_type_id)( be_it );

            if( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t
                    )NS(Object_get_begin_ptr)( be_it );

                if( monitor != SIXTRL_NULLPTR )
                {
                    NS(BeamMonitor_clear)( monitor );
                }
            }
        }
    }
    else
    {
        if( beam_elements == SIXTRL_NULLPTR )
        {
            flags |= BELEM_BUFFER_NULL_FLAG;
        }

        if( slot_size == ( buf_size_t )0u )
        {
            flags |= SLOT_SIZE_ILLEGAL_FLAG;
        }

        if( NS(ManagedBuffer_needs_remapping)( beam_elements, slot_size ) )
        {
            flags |= BELEM_BUFFER_REQUIRES_REMAP_FLAG;
        }

        if( NS(ManagedBuffer_get_num_objects)( beam_elements, slot_size ) ==
            ZERO )
        {
            flags |= BELEM_BUFFER_EMPTY_FLAG;
        }
    }

    if( ptr_status_flags != SIXTRL_NULLPTR ) *ptr_status_flags = flags;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_C99_HEADER_H__ */

/* end: sixtracklib/common/be_monitor/be_monitor.h */
