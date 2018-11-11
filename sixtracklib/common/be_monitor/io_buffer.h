#ifndef SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_BE_MONITOR_IO_BUFFER_C99_H__
#define SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_BE_MONITOR_IO_BUFFER_C99_H__


#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( _GPUCODE )

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_elem_by_elem_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects)( SIXTRL_BUFFER_ARGPTR_DEC const
        NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer );

SIXTRL_HOST_FN int NS(BeamMonitor_prepare_io_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT belements,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const num_elem_by_elem_turns );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_assign_io_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const num_elem_by_elem_turns  );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_assign_io_buffer_from_offset)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const io_particles_block_offset );

#endif /* !defined( _GPUCODE ) */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t)
NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC void NS(BeamMonitor_clear_all_on_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const slot_size );

SIXTRL_FN SIXTRL_STATIC int NS(BeamMonitor_assign_managed_io_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const io_particles_block_offset,
    NS(buffer_size_t) const slot_size );

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

#if !defined( _GPUCODE )

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

SIXTRL_INLINE int NS(BeamMonitor_assign_io_buffer)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const slot_size = NS(Buffer_get_slot_size)(
        beam_elements_buffer );

    buf_size_t const max_num_available_io_blocks =
        NS(Particles_managed_buffer_get_num_of_particle_blocks)(
            NS(Buffer_get_const_data_begin)( io_buffer ), slot_size );

    buf_size_t io_particles_block_offset = ( buf_size_t )0u;
    SIXTRL_ASSERT( slot_size == NS(Buffer_get_slot_size)( io_buffer ) );

    if( num_elem_by_elem_turns > ( buf_size_t )0u )
    {
        buf_size_t const num_elem_by_elem_blocks =
            NS(BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer)(
                NS(Buffer_get_const_data_begin)( beam_elements_buffer ),
                    slot_size );

        buf_size_t const requ_num_elem_by_elem_blocks =
            num_elem_by_elem_turns * num_elem_by_elem_blocks;

        if( requ_num_elem_by_elem_blocks <= max_num_available_io_blocks )
        {
            io_particles_block_offset = requ_num_elem_by_elem_blocks;
        }
    }

    return NS(BeamMonitor_assign_managed_io_buffer)(
        NS(Buffer_get_data_begin)( beam_elements_buffer ),
        NS(Buffer_get_data_begin)( io_buffer ), num_particles,
        io_particles_block_offset, slot_size );
}

SIXTRL_INLINE int NS(BeamMonitor_assign_io_buffer_from_offset)(
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const io_particles_block_offset )
{
    return NS(BeamMonitor_assign_managed_io_buffer)(
        NS(Buffer_get_data_begin)( beam_elements_buffer ),
        NS(Buffer_get_data_begin)( io_buffer ), num_particles,
        io_particles_block_offset, NS(Buffer_get_slot_size)( io_buffer ) );
}

#endif /* !defined( _GPUCODE ) */


SIXTRL_INLINE NS(buffer_size_t)
NS(BeamMonitor_get_num_elem_by_elem_objects_from_managed_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    buf_size_t num_elem_by_elem_blocks = ZERO;

    if( ( beam_elements != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( NS(ManagedBuffer_get_num_objects)( beam_elements, slot_size ) ) > ZERO )
    {
        ptr_obj_t be_it  = NS(ManagedBuffer_get_const_objects_index_begin)(
            beam_elements, slot_size );

        ptr_obj_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            beam_elements, slot_size );

        for( ; be_it != be_end ; ++be_it )
        {
            NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_it );

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
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const slot_size )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;
    buf_size_t num_beam_monitors = ZERO;

    if( ( beam_elements != SIXTRL_NULLPTR ) && ( slot_size > ZERO ) &&
        ( NS(ManagedBuffer_get_num_objects)( beam_elements, slot_size ) ) > ZERO )
    {
        ptr_obj_t be_it  = NS(ManagedBuffer_get_const_objects_index_begin)(
            beam_elements, slot_size );

        ptr_obj_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
            beam_elements, slot_size );

        for( ; be_it != be_end ; ++be_it )
        {
            NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_it );

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
        ( NS(ManagedBuffer_get_num_objects)( beam_elements, slot_size ) ) > ZERO )
    {
        ptr_obj_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
            beam_elements, slot_size );

        ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            beam_elements, slot_size );

        for( ; be_it != be_end ; ++be_it )
        {
            NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_it );

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


SIXTRL_INLINE int NS(BeamMonitor_assign_managed_io_buffer)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT beam_elements,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT io_buffer,
    NS(buffer_size_t) const num_particles,
    NS(buffer_size_t) const io_particles_block_offset,
    NS(buffer_size_t) const slot_size )
{
    int success = -1;

    typedef NS(buffer_size_t) buf_size_t;
    typedef NS(be_monitor_turn_t) nturn_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)* ptr_obj_t;
    typedef SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)* ptr_beam_monitor_t;
    typedef NS(buffer_addr_t) addr_t;

    SIXTRL_STATIC_VAR buf_size_t const ZERO = ( buf_size_t )0u;

    buf_size_t const num_io_particle_blocks =
        NS(Particles_managed_buffer_get_num_of_particle_blocks)(
            io_buffer, slot_size );

    buf_size_t num_io_particle_blocks_assigned = ZERO;

    buf_size_t const num_beam_monitors =
        NS(BeamMonitor_get_num_of_beam_monitor_objects_from_managed_buffer)(
            beam_elements, slot_size );

    ptr_obj_t io_it = NS(ManagedBuffer_get_objects_index_begin)(
        io_buffer, slot_size );

    ptr_obj_t io_end = NS(ManagedBuffer_get_objects_index_end)(
        io_buffer, slot_size );

    ( void )io_end;

    SIXTRL_ASSERT( io_it  != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( io_end != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( ( intptr_t )io_end - ( intptr_t )io_it ) >=
                     ( intptr_t )num_io_particle_blocks );

    success = ( ( io_particles_block_offset + num_beam_monitors ) <=
                 num_io_particle_blocks ) ? 0 : -1;

    if( ( success == 0 ) && ( num_beam_monitors > ZERO ) )
    {
        buf_size_t num_beam_monitors_assigned = ZERO;

        buf_size_t const io_store_stride =
            NS(Particles_get_required_num_slots_on_managed_buffer)(
                num_particles, slot_size ) * slot_size;

        ptr_obj_t be_it  = NS(ManagedBuffer_get_objects_index_begin)(
            beam_elements, slot_size );

        ptr_obj_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            beam_elements, slot_size );

        SIXTRL_ASSERT( io_store_stride >= sizeof( NS(Particles) ) );

        io_it = io_it + io_particles_block_offset;

        SIXTRL_ASSERT( be_it  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_end != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_it  != be_end );

        SIXTRL_ASSERT( ( ( intptr_t )be_end - ( intptr_t )be_it ) >=
            ( intptr_t )NS(ManagedBuffer_get_num_objects)(
                beam_elements, slot_size ) );

        NS(ManagedBuffer_get_num_objects)( beam_elements, slot_size );

        for( ; be_it != be_end ; ++be_it )
        {
            NS(object_type_id_t) const type_id = NS(Object_get_type_id)( be_it );
            uintptr_t const addr = ( uintptr_t )NS(Object_get_begin_addr)( be_it );

            if( ( type_id == NS(OBJECT_TYPE_BEAM_MONITOR) ) && ( addr != ZERO ) )
            {
                ptr_beam_monitor_t monitor = ( ptr_beam_monitor_t )addr;
                nturn_t const nn = NS(BeamMonitor_get_num_stores)( monitor );

                if( nn > 0 )
                {
                    if( ( num_io_particle_blocks_assigned + nn ) <=
                          num_io_particle_blocks )
                    {
                        addr_t const paddr = ( addr_t )( uintptr_t
                            )NS(Object_get_begin_addr)( io_it );

                        #if !defined( NDEBUG )
                        nturn_t kk = ( nturn_t )0;

                        for( ; kk < nn ; ++kk )
                        {
                            SIXTRL_ASSERT( NS(Object_get_begin_addr)(
                                io_it + kk ) == ( addr_t )(
                                    paddr + kk * io_store_stride ) );
                        }
                        #endif /* !defined( NDEBUG ) */

                        NS(BeamMonitor_set_io_address)( monitor, paddr );
                        monitor->io_store_stride = io_store_stride;

                        io_it = io_it + nn;
                        num_io_particle_blocks_assigned += nn;

                        SIXTRL_ASSERT( ZERO <= (
                            ( ( uintptr_t )io_end ) - ( uintptr_t )io_it ) );
                    }
                    else
                    {
                        success = -1;
                        break;
                    }
                }
                else
                {
                    NS(BeamMonitor_clear)( monitor );
                }

                ++num_beam_monitors_assigned;

                if( num_beam_monitors_assigned == num_beam_monitors )
                {
                    break;
                }
            }

            if( success != 0 ) break;
        }
    }

    return success;
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_BE_MONITOR_IO_BUFFER_C99_H__ */

/* end: sixtracklib/common/be_monitor/be_monitor_io_buffer.h */
