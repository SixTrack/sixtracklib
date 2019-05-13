#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/cuda/kernels/track_particles_kernels.cuh"
#endif /* !defined( SIXTRL_NO_INCLUDES )     */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>

    #include <cuda_runtime_api.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"

    #include "sixtracklib/cuda/cuda_tools.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__global__ void NS(Track_particles_until_turn_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particles_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;

    nelements_t particle_index = NS(Cuda_get_1d_thread_id)();
    nelements_t const stride = NS(Cuda_get_total_num_threads)();

    ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        pbuffer_begin, pset_index, slot_size );

    nelements_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    be_iter_t belem_begin = NS(ManagedBuffer_get_objects_index_begin)(
        be_buffer_begin, slot_size );

    be_iter_t belem_end = NS(ManagedBuffer_get_objects_index_end)(
        be_buffer_end, slot_size );

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
    SIXTRL_DATAPTR_DEC NS(ctrl_debug_flag_t)* SIXTRL_RESTRICT ptr_status_flg )
{
    typedef NS(particles_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(ctrl_debug_flag_t) debug_flag_t;

    debug_flag_t debug_flag    = SIXTRL_CONTROLLER_DEBUG_FLAG_OK;
    nelements_t particle_index = NS(Cuda_get_1d_thread_id)();
    nelements_t const stride   = NS(Cuda_get_total_num_threads)();

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
            be_buffer_end, slot_size );

        if( ( particles != SIXTRL_NULLPTR ) &&
            ( be_begin != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) )
        {
            track_status_t track_status = NS(TRACK_SUCCESS);

            while( ( track_status == NS(TRACK_SUCCESS) ) &&
                   ( particle_index < num_particles ) )
            {
                track_status = NS(Track_particle_until_turn_obj)( particles,
                    particle_index, be_begin, be_end, until_turn );

                particle_index += stride;
            }

            if( track_status != NS(TRACK_SUCCESS) )
            {
                if( track_status < ( track_status_t) )
                {
                    track_status = -track_status;
                }

                debug_flag |= ( debug_flag_t )track_status;
            }
        }
        else
        {
            debug_flag |= ( debug_flag_t )0x1000;
        }
    }
    else
    {
        debug_flag |= ( debug_flag_t )0x2000;
    }

    NS(Cuda_handle_debug_flag_in_kernel)( ptr_status_flg, debug_flag );
}

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_track_elem_by_elem_until_turn_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const until_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particles_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;

    nelements_t particle_index = NS(Cuda_get_1d_thread_id)();
    nelements_t const stride = NS(Cuda_get_total_num_threads)();

    ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        pbuffer_begin, pset_index, slot_size );

    nelements_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    be_iter_t belem_begin = NS(ManagedBuffer_get_objects_index_begin)(
        be_buffer_begin, slot_size );

    be_iter_t belem_end = NS(ManagedBuffer_get_objects_index_end)(
        be_buffer_end, slot_size );

    for( ; particle_index < num_particles ; particle_index += stride )
    {
        NS(Track_particle_element_by_element_until_turn_objs)( particles,
            particle_index, elem_by_elem_config, belem_begin, belem_end,
                until_turn );
    }
}

__global__ void NS(Track_track_elem_by_elem_until_turn_kernel_cuda_debug)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT output_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC const NS(ElemByElemConfig) *const
        SIXTRL_RESTRICT elem_by_elem_config,
    NS(buffer_size_t) const out_buffer_offset_index,
    NS(buffer_size_t) const until_turn,
    NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(ctrl_debug_flag_t)* SIXTRL_RESTRICT ptr_status_flg )
{
    typedef NS(particles_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(ctrl_debug_flag_t) debug_flag_t;
    typedef NS(elem_by_elem_out_addr_t) e_by_e_out_addr_t;

    debug_flag_t debug_flag    = SIXTRL_CONTROLLER_DEBUG_FLAG_OK;
    nelements_t particle_index = NS(Cuda_get_1d_thread_id)();
    nelements_t const stride   = NS(Cuda_get_total_num_threads)();

    if( ( pbuffer_begin != SIXTRL_NULLPTR ) &&
        ( be_buffer_begin != SIXTRL_NULLPTR ) &&
        ( slot_size > ( NS(buffer_size_t) )0u ) &&
        ( ouput_buffer != SIXTRL_NULLPTR ) &&
        ( elem_by_elem_config != SIXTRL_NULLPTR ) &&
        ( NS(ElemByElemConfig_get_output_store_address)(
            elem_by_elem_config ) > ( e_by_e_out_addr_t )0u ) &&
        ( out_buffer_offset_index < NS(ManagedBuffer_get_num_objects)(
            output_buffer, slot_size ) ) &&
    {
        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            pbuffer_begin, pset_index, slot_size );

        nelements_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        be_iter_t be_begin = NS(ManagedBuffer_get_objects_index_begin)(
            be_buffer_begin, slot_size );

        be_iter_t be_end = NS(ManagedBuffer_get_objects_index_end)(
            be_buffer_end, slot_size );

        if( ( particles != SIXTRL_NULLPTR ) &&
            ( be_begin != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) )
        {
            track_status_t track_status = NS(TRACK_SUCCESS);

            while( ( track_status == NS(TRACK_SUCCESS) ) &&
                   ( particle_index < num_particles ) )
            {
                track_status =
                NS(Track_particle_element_by_element_until_turn_objs)(
                    particles, particle_index, elem_by_elem_config,
                        be_begin, be_end, until_turn );

                particle_index += stride;
            }

            if( track_status != NS(TRACK_SUCCESS) )
            {
                if( track_status < ( track_status_t) )
                {
                    track_status = -track_status;
                }

                debug_flag |= ( debug_flag_t )track_status;
            }
        }
        else
        {
            debug_flag |= ( debug_flag_t )0x1000;
        }
    }
    else if( ( elem_by_elem_config == SIXTRL_NULLPTR ) ||
             ( NS(ElemByElemConfig_get_output_store_address)(
                  elem_by_elem_config ) == ( e_by_e_out_addr_t )0u ) )
    {
        debug_flag |= ( debug_flag_t )0x2000;
    }
    else
    {
        debug_flag |= ( debug_flag_t )0x4000;
    }

    NS(Cuda_handle_debug_flag_in_kernel)( ptr_status_flg, debug_flag );
}

/* ------------------------------------------------------------------------- */

__global__ void NS(Track_particles_line_kernel_cuda)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size )
{
    typedef NS(particles_num_elements_t) nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;

    nelements_t particle_index = NS(Cuda_get_1d_thread_id)();
    nelements_t const stride = NS(Cuda_get_total_num_threads)();

    ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
        pbuffer_begin, pset_index, slot_size );

    nelements_t const num_particles =
        NS(Particles_get_num_of_particles)( particles );

    for( ; particle_index < num_particles ; particle_index += stride )
    {
        NS(Track_particle_line)( particles, particle_index, belem_begin_id,
            belem_end_id, finish_turn );
    }
}

__global__ void NS(CudaTrack_particles_line_debug_kernel)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer_begin,
    NS(buffer_size_t) const pset_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT be_buffer_begin,
    NS(buffer_size_t) const belem_begin_id,
    NS(buffer_size_t) const belem_end_id,
    bool const finish_turn, NS(buffer_size_t) const slot_size,
    SIXTRL_DATAPTR_DEC NS(ctrl_debug_flag_t)* SIXTRL_RESTRICT ptr_status_flg )
{
    typedef NS(particles_num_elements_t)    nelements_t;
    typedef SIXTRL_DATAPTR_DEC NS(Particles)* ptr_particles_t;
    typedef SIXTRL_DATAPTR_DEC NS(Object) const* be_iter_t;
    typedef NS(track_status_t) track_status_t;
    typedef NS(ctrl_debug_flag_t) debug_flag_t;

    debug_flag_t debug_flag    = SIXTRL_CONTROLLER_DEBUG_FLAG_OK;
    nelements_t particle_index = NS(Cuda_get_1d_thread_id)();
    nelements_t const stride   = NS(Cuda_get_total_num_threads)();

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
            be_buffer_end, slot_size );

        if( ( particles != SIXTRL_NULLPTR ) &&
            ( be_begin != SIXTRL_NULLPTR ) && ( be_end != SIXTRL_NULLPTR ) )
        {
            track_status_t track_status = NS(TRACK_SUCCESS);

            while( ( track_status == NS(TRACK_SUCCESS) ) &&
                   ( particle_index < num_particles ) )
            {
                track_status = NS(Track_particle_until_turn_obj)( particles,
                    particle_index, be_begin, be_end, until_turn );

                particle_index += stride;
            }

            if( track_status != NS(TRACK_SUCCESS) )
            {
                if( track_status < ( track_status_t) )
                {
                    track_status = -track_status;
                }

                debug_flag |= ( debug_flag_t )track_status;
            }
        }
        else
        {
            debug_flag |= ( debug_flag_t )0x1000;
        }
    }
    else
    {
        debug_flag |= ( debug_flag_t )0x2000;
    }

    NS(Cuda_handle_debug_flag_in_kernel)( ptr_status_flg, debug_flag );
}

/* end sixtracklib/cuda/kernels/track_particles_kernels.cu */
