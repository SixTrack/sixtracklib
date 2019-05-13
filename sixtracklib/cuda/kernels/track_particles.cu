#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/track_particles.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES )     */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"

    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Track_particles_until_turn_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;

    nelements_t particle_index = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const stride = NS(Cuda_get_total_num_threads_in_kernel)();

    ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        pbuffer_begin, pset_index, slot_size );

    nelements_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    be_iter_t belem_begin = NS(ManagedBuffer_get_objects_index_begin)(
        be_buffer_begin, slot_size );

    be_iter_t belem_end = NS(ManagedBuffer_get_objects_index_end)(
        be_buffer_begin, slot_size );

    for( ; particle_index < num_particles ; particle_index += stride )
    {
        NS(Track_particle_until_turn_obj)(
            particles, particle_index, belem_begin, belem_end, until_turn );
    }
}

__global__ void NS(Track_particles_until_turn_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(arch_status_t) arch_status_t;
    typedef NS(arch_debugging_t) dbg_t;

    dbg_t dbg = SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;
    nelements_t particle_index = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const stride   = NS(Cuda_get_total_num_threads_in_kernel)();

    if( ( pbuffer_begin != SIXTRL_NULLPTR ) &&
        ( be_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            pbuffer_begin, pset_index, slot_size );

        nelements_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        be_iter_t be_begin = NS(ManagedBuffer_get_objects_index_begin)(
            be_buffer_begin, slot_size );

        be_iter_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            be_buffer_begin, slot_size );

        if( ( particles != SIXTRL_NULLPTR ) &&
            ( be_begin != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) )
        {
            track_status_t track_status = SIXTRL_TRACK_SUCCESS;

            while( ( track_status == SIXTRL_TRACK_SUCCESS ) &&
                   ( particle_index < num_particles ) )
            {
                track_status = NS(Track_particle_until_turn_obj)( particles,
                    particle_index, be_begin, be_end, until_turn );

                particle_index += stride;
            }

            if( track_status != SIXTRL_TRACK_SUCCESS )
            {
                dbg = NS(DebugReg_store_arch_status)(
                    dbg, ( arch_status_t )track_status );
            }
        }
        else
        {
            dbg = NS(DebugReg_raise_next_error_flag)( dbg );
        }
    }
    else
    {
        dbg = NS(DebugReg_raise_next_error_flag)( dbg );
    }

    if( ( NS(DebugReg_has_any_flags_set)( dbg ) ) &&
        ( ptr_dbg_register != SIXTRL_NULLPTR ) )
    {
        *ptr_dbg_register |= dbg;
    }
}

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_track_elem_by_elem_until_turn_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;

    nelements_t particle_index = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const stride = NS(Cuda_get_total_num_threads_in_kernel)();

    ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        pbuffer_begin, pset_index, slot_size );

    nelements_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    be_iter_t belem_begin = NS(ManagedBuffer_get_objects_index_begin)(
        be_buffer_begin, slot_size );

    be_iter_t belem_end = NS(ManagedBuffer_get_objects_index_end)(
        be_buffer_begin, slot_size );

    for( ; particle_index < num_particles ; particle_index += stride )
    {
        NS(Track_particle_element_by_element_until_turn_objs)( particles,
            particle_index, elem_by_elem_config, belem_begin, belem_end,
                until_turn );
    }
}

__global__ void NS(Track_track_elem_by_elem_until_turn_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    typedef NS(particle_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(arch_status_t) arch_status_t;
    typedef NS(arch_debugging_t) dbg_t;
    typedef NS(elem_by_elem_out_addr_t) e_by_e_out_addr_t;

    dbg_t dbg    = SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;
    nelements_t particle_index = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const stride   = NS(Cuda_get_total_num_threads_in_kernel)();

    if( ( pbuffer_begin != SIXTRL_NULLPTR ) &&
        ( be_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) &&
        ( output_buffer != SIXTRL_NULLPTR ) &&
        ( elem_by_elem_config != SIXTRL_NULLPTR ) &&
        ( NS(ElemByElemConfig_get_output_store_address)(
            elem_by_elem_config ) > ( e_by_e_out_addr_t )0u ) &&
        ( out_buffer_offset_index < NS(ManagedBuffer_get_num_objects)(
            output_buffer, slot_size ) ) )
    {
        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            pbuffer_begin, pset_index, slot_size );

        nelements_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        be_iter_t be_begin = NS(ManagedBuffer_get_objects_index_begin)(
            be_buffer_begin, slot_size );

        be_iter_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            be_buffer_begin, slot_size );

        if( ( particles != SIXTRL_NULLPTR ) &&
            ( be_begin != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) )
        {
            track_status_t track_status = SIXTRL_TRACK_SUCCESS;

            while( ( track_status == SIXTRL_TRACK_SUCCESS ) &&
                   ( particle_index < num_particles ) )
            {
                track_status =
                NS(Track_particle_element_by_element_until_turn_objs)(
                    particles, particle_index, elem_by_elem_config,
                        be_begin, be_end, until_turn );

                particle_index += stride;
            }

            if( track_status != SIXTRL_TRACK_SUCCESS )
            {
                dbg = NS(DebugReg_store_arch_status)(
                    dbg, ( arch_status_t )track_status );
            }
        }
        else
        {
             dbg = NS(DebugReg_raise_next_error_flag)( dbg );
        }
    }
    else if( ( elem_by_elem_config == SIXTRL_NULLPTR ) ||
             ( NS(ElemByElemConfig_get_output_store_address)(
                  elem_by_elem_config ) == ( e_by_e_out_addr_t )0u ) )
    {
         dbg = NS(DebugReg_raise_next_error_flag)( dbg );
    }
    else
    {
         dbg = NS(DebugReg_raise_next_error_flag)( dbg );
    }

    if( ( NS(DebugReg_has_any_flags_set)( dbg ) ) &&
        ( ptr_dbg_register != SIXTRL_NULLPTR ) )
    {
        *ptr_dbg_register |= dbg;
    }
}

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_particles_line_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particle_num_elements_t) nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_elem_iter_t;

    nelements_t pidx = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const stride = NS(Cuda_get_total_num_threads_in_kernel)();

    ptr_particles_t p = NS(Particles_managed_buffer_get_particles)(
        pbuffer_begin, pset_index, slot_size );

    nelements_t const num_particles = NS(Particles_get_num_of_particles)( p );
    be_elem_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        be_buffer_begin, slot_size );

    be_elem_iter_t be_end = be_begin;

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( belem_begin_id <= belem_end_id );
    SIXTRL_ASSERT( belem_end_id <= NS(ManagedBuffer_get_num_objects)(
        be_buffer_begin, slot_size ) );

    be_begin = be_begin + belem_begin_id;
    be_end   = be_end   + belem_end_id;

    for( ; pidx < num_particles ; pidx += stride )
    {
        NS(Track_particle_line)( p, pidx, be_begin, be_end, finish_turn );
    }
}

__global__ void NS(Track_particles_line_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(arch_debugging_t)* SIXTRL_RESTRICT ptr_dbg_register )
{
    typedef NS(particle_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_elem_iter_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(arch_status_t) arch_status_t;
    typedef NS(arch_debugging_t) dbg_t;

    dbg_t dbg = SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;
    nelements_t particle_index = NS(Cuda_get_1d_thread_id_in_kernel)();
    nelements_t const stride   = NS(Cuda_get_total_num_threads_in_kernel)();

    if( ( pbuffer_begin != SIXTRL_NULLPTR ) &&
        ( be_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) )
    {
        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            pbuffer_begin, pset_index, slot_size );

        nelements_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        be_elem_iter_t be_begin = NS(ManagedBuffer_get_objects_index_begin)(
            be_buffer_begin, slot_size );

        be_elem_iter_t be_end = be_begin;

        if( ( particles != SIXTRL_NULLPTR ) && ( be_begin != SIXTRL_NULLPTR )
            && ( belem_begin_id <= belem_end_id ) &&
            ( belem_end_id <= NS(ManagedBuffer_get_num_objects)(
                be_buffer_begin, slot_size ) ) )
        {
            track_status_t track_status = SIXTRL_TRACK_SUCCESS;

            be_begin = be_begin + belem_begin_id;
            be_end = be_end + belem_end_id;

            while( ( track_status == SIXTRL_TRACK_SUCCESS ) &&
                   ( particle_index < num_particles ) )
            {
                track_status = NS(Track_particle_line)( particles,
                    particle_index, be_begin, be_end, finish_turn );

                particle_index += stride;
            }

            if( track_status != SIXTRL_TRACK_SUCCESS )
            {
                dbg = NS(DebugReg_store_arch_status)(
                    dbg, ( arch_status_t )track_status );
            }
        }
        else
        {
             dbg = NS(DebugReg_raise_next_error_flag)( dbg );
        }
    }
    else
    {
         dbg = NS(DebugReg_raise_next_error_flag)( dbg );
    }

    if( ( NS(DebugReg_has_any_flags_set)( dbg ) ) &&
        ( ptr_dbg_register != SIXTRL_NULLPTR ) )
    {
        *ptr_dbg_register |= dbg;
    }
}

/* end sixtracklib/cuda/kernels/track_particles_kernels.cu */
