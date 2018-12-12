#ifndef SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_C99_H__
#define SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_C99_H__


#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/internal/elem_by_elem_config_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */
/* BeamMonitor based Output: */

struct NS(BeamMonitor);

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_setup_for_particles)(
    SIXTRL_BE_ARGPTR_DEC struct NS(BeamMonitor)* SIXTRL_RESTRICT beam_monitor,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

SIXTRL_FN SIXTRL_STATIC int
NS(BeamMonitor_setup_managed_buffer_for_particles_all)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const slot_size );

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC bool
NS(BeamMonitor_are_present_in_buffer)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT belements_buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_elem_by_elem_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT belements_buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT belements_buffer );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(BeamMonitor_setup_for_particles_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p );

SIXTRL_EXTERN SIXTRL_HOST_FN int NS(BeamMonitor_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const num_elem_by_elem_turns );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_assign_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns  );

SIXTRL_FN SIXTRL_STATIC int
NS(BeamMonitor_assign_particles_out_buffer_from_offset)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const out_particles_block_offset );

#endif /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT
        belements_buffer, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT
        blements_buffer, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT
        beam_elements_buffer, NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int
NS(BeamMonitor_assign_managed_particles_out_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT belements_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const out_index, NS(buffer_size_t) const slot_size );

/* ------------------------------------------------------------------------- */
/* Element - by - Element Output: */

struct NS(ElemByElemConfig);

#if !defined( _GPUCODE )

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_prepare_particles_out_buffer_detailed)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        struct NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT
        ptr_out_buffer_index_offset );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_prepare_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const SIXTRL_RESTRICT p,
    NS(buffer_size_t) const num_elem_by_elem_turns );

SIXTRL_EXTERN SIXTRL_HOST_FN int
NS(ElemByElemConfig_assign_particles_out_buffer)(
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
        struct NS(ElemByElemConfig)* SIXTRL_RESTRICT config,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const out_buffer_index_offset );

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

    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */
/* BeamMonitor based Output */

#if !defined( _GPUCODE )

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
NS(BeamMonitor_get_num_elem_by_elem_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer )
{
    return NS(BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer)(
        NS(Buffer_get_const_data_begin)( beam_elements_buffer ),
        NS(Buffer_get_slot_size)( beam_elements_buffer ) );
}

SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer )
{
    return NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
        NS(Buffer_get_const_data_begin)( beam_elements_buffer ),
        NS(Buffer_get_slot_size)( beam_elements_buffer ) );
}

SIXTRL_INLINE void NS(BeamMonitor_clear_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer )
{
    NS(BeamMonitor_clear_all_on_managed_buffer)(
        NS(Buffer_get_data_begin)( beam_elements_buffer),
        NS(Buffer_get_slot_size)( beam_elements_buffer ) );

    return;
}

SIXTRL_INLINE int NS(BeamMonitor_setup_for_particles)(
    SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* SIXTRL_RESTRICT beam_monitor,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles) *const
        SIXTRL_RESTRICT particles )
{
    int success = -1;

    typedef NS(particle_num_elements_t) num_elements_t;
    typedef NS(particle_index_t)        index_t;
    typedef NS(be_monitor_addr_t)       addr_t;

    num_elements_t const num_particles = NS(Particles_get_num_of_particles)(
        particles );

    if( ( beam_monitor != SIXTRL_NULLPTR ) &&
        ( particles != SIXTRL_NULLPTR ) &&
        ( num_particles > ( num_elements_t )0u ) )
    {
        SIXTRL_STATIC_VAR index_t const ZERO = ( index_t )0u;

        num_elements_t ii = 0;
        success = 0;

        index_t min_particle_id = ZERO;
        index_t max_particle_id = ZERO;

        index_t particle_id = NS(Particles_get_particle_id_value)(
            particles, ii++ );

        if( particle_id < ZERO ) particle_id = -particle_id;

        min_particle_id = max_particle_id = particle_id;

        for( ; ii < num_particles ; ++ii )
        {
            particle_id = NS(Particles_get_particle_id_value)( particles, ii );
            if( particle_id < ZERO ) particle_id = -particle_id;

            if( min_particle_id > particle_id ) min_particle_id = particle_id;
            if( max_particle_id < particle_id ) max_particle_id = particle_id;
        }

        ( void )min_particle_id;
        min_particle_id = ZERO;

        NS(BeamMonitor_set_min_particle_id)( beam_monitor, min_particle_id );
        NS(BeamMonitor_set_max_particle_id)( beam_monitor, max_particle_id );
        NS(BeamMonitor_set_out_address)( beam_monitor, ( addr_t )0u );

        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(BeamMonitor_setup_managed_buffer_for_particles_all)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_PARTICLE_ARGPTR_DEC const NS(Particles)
        *const SIXTRL_RESTRICT particles,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(particle_num_elements_t) num_elements_t;
    typedef NS(particle_index_t)        index_t;
    typedef NS(be_monitor_addr_t)       addr_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*    ptr_beam_monitor_t;

    num_elements_t const num_particles = NS(Particles_get_num_of_particles)(
        particles );

    ptr_obj_t be_it = NS(ManagedBuffer_get_objects_index_begin)(
            beam_elements_buffer, slot_size );

    ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            beam_elements_buffer, slot_size );

    if( ( be_it != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) &&
        ( slot_size != ( NS(buffer_size_t) )0u ) &&
        ( num_particles > ( num_elements_t )0u ) )
    {
        SIXTRL_STATIC_VAR index_t const ZERO = ( index_t )0u;

        num_elements_t ii = 0;
        success = 0;

        index_t min_particle_id = ZERO;
        index_t max_particle_id = ZERO;

        index_t particle_id = NS(Particles_get_particle_id_value)(
            particles, ii++ );

        SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
            beam_elements_buffer, slot_size ) );

        SIXTRL_ASSERT( ( uintptr_t )be_end >= ( uintptr_t )be_it );

        if( particle_id < ZERO ) particle_id = -particle_id;

        min_particle_id = max_particle_id = particle_id;

        for( ; ii < num_particles ; ++ii )
        {
            particle_id = NS(Particles_get_particle_id_value)( particles, ii );
            if( particle_id < ZERO ) particle_id = -particle_id;

            if( min_particle_id > particle_id ) min_particle_id = particle_id;
            if( max_particle_id < particle_id ) max_particle_id = particle_id;
        }

        ( void )min_particle_id;
        min_particle_id = ZERO;

        for( ; be_it != be_end ; ++be_it )
        {
            if( NS(Object_get_type_id)( be_it ) ==
                NS(OBJECT_TYPE_BEAM_MONITOR) )
            {
                ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t
                    )NS(Object_get_begin_ptr)( be_it );

                NS(BeamMonitor_set_min_particle_id)( monitor, min_particle_id );
                NS(BeamMonitor_set_max_particle_id)( monitor, max_particle_id );
                NS(BeamMonitor_set_out_address)( monitor, ( addr_t )0 );
            }
        }

        success = 0;
    }

    return success;
}

SIXTRL_INLINE int NS(BeamMonitor_assign_particles_out_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)(
        beam_elements_buffer );

    buf_size_t out_particles_block_offset = ( buf_size_t )0u;
    SIXTRL_ASSERT( slot_size == NS(Buffer_get_slot_size)( out_buffer ) );

    if( ( num_elem_by_elem_turns > ( buf_size_t )0u ) &&
        (  NS(BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer)(
                NS(Buffer_get_const_data_begin)( beam_elements_buffer ),
                    slot_size ) > ( buf_size_t )0u ) )
    {
        out_particles_block_offset = ( buf_size_t )1u;
    }

    return NS(BeamMonitor_assign_managed_particles_out_buffer)(
        NS(Buffer_get_data_begin)( beam_elements_buffer ),
        NS(Buffer_get_data_begin)( out_buffer ), out_particles_block_offset,
        slot_size );
}

SIXTRL_INLINE int NS(BeamMonitor_assign_particles_out_buffer_from_offset)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const out_particles_block_offset )
{
    return NS(BeamMonitor_assign_managed_particles_out_buffer)(
        NS(Buffer_get_data_begin)( beam_elements_buffer ),
        NS(Buffer_get_data_begin)( out_buffer ),
        out_particles_block_offset, NS(Buffer_get_slot_size)( out_buffer ) );
}

#endif /* !defined( _GPUCODE ) */


SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT beam_elements, NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    buf_size_t num_elem_by_elem_blocks = ZERO;

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

            if( ( type_id != NS(OBJECT_TYPE_NONE)         ) &&
                ( type_id != NS(OBJECT_TYPE_PARTICLE)     ) &&
                ( type_id != NS(OBJECT_TYPE_LINE)         ) &&
                ( type_id != NS(OBJECT_TYPE_INVALID)      ) )
            {
                ++num_elem_by_elem_blocks;
            }
        }
    }

    return num_elem_by_elem_blocks;
}

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
                ( NS(Object_get_const_begin_ptr)( be_it ) != SIXTRL_NULLPTR ) &&
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


SIXTRL_INLINE int NS(BeamMonitor_assign_managed_particles_out_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT out_buffer,
    NS(buffer_size_t) const out_particles_block_offset,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(buffer_size_t)                               buf_size_t;
    typedef NS(be_monitor_turn_t)                           nturn_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*        ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*           ptr_beam_monitor_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(Particles) const*  ptr_particles_t;
    typedef NS(be_monitor_index_t)                          mon_index_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    ptr_obj_t out_it = NS(ManagedBuffer_get_objects_index_begin)(
        out_buffer, slot_size );

    ptr_obj_t out_end = NS(ManagedBuffer_get_objects_index_end)(
        out_buffer, slot_size );

    ptr_obj_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
        beam_elements, slot_size );

    ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
        beam_elements, slot_size );

    SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( out_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( out_end != SIXTRL_NULLPTR );

    out_it = out_it + out_particles_block_offset;

    if( ( uintptr_t )out_end >= ( uintptr_t )out_it )
    {
        success = 0;
    }
    else
    {
        return success;
    }

    for( ; be_it != be_end ; ++be_it )
    {
        NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_it );
        uintptr_t const addr = ( uintptr_t )NS(Object_get_begin_addr)( be_it );

        if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) ) && ( addr != ZERO ) )
        {
            ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t )addr;
            nturn_t const nn = NS(BeamMonitor_get_num_stores)( monitor );

            if( ( nn > ( nturn_t )0u ) && ( out_it != out_end ) )
            {
                ptr_particles_t particles = ( ptr_particles_t
                    )NS(BufferIndex_get_const_particles)( out_it );

                buf_size_t const num_stored_particles =
                    NS(Particles_get_num_of_particles)( particles );

                mon_index_t const min_particle_id =
                    NS(BeamMonitor_get_min_particle_id)( monitor );

                mon_index_t const max_particle_id =
                    NS(BeamMonitor_get_max_particle_id)( monitor );

                buf_size_t const stored_particles_per_turn =
                    ( max_particle_id >= min_particle_id )
                        ? ( buf_size_t )(
                            max_particle_id - min_particle_id  + 1u )
                        : ZERO;

                if( ( nn > 0 ) && ( stored_particles_per_turn > ZERO ) &&
                    ( particles != SIXTRL_NULLPTR ) &&
                    ( ( stored_particles_per_turn * ( buf_size_t )nn ) <=
                        num_stored_particles ) )
                {
                    NS(BeamMonitor_set_out_address)(
                        monitor, NS(Object_get_begin_addr)( out_it++ ) );
                }
            }
            else if( nn > ( nturn_t )0u )
            {
                success = -1;
                break;
            }
        }
    }

    return success;
}

/* ------------------------------------------------------------------------- */
/* Element - by - Element Output: */



#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_C99_H__ */

/* end: sixtracklib/common/output/output_buffer.h */
