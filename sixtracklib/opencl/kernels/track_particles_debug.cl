#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"
    #include "sixtracklib/opencl/internal/success_flag.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(Track_particles_line_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_UINT64_T const particle_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_UINT64_T const line_begin_idx, SIXTRL_UINT64_T const line_end_idx,
    SIXTRL_UINT64_T const finish_turn,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
        SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_single_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const increment_turn,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
        SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_until_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const until_turn,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
        SIXTRL_RESTRICT ptr_success_flag );

__kernel void NS(Track_particles_elem_by_elem_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_INT64_T const until_turn,
    SIXTRL_UINT64_T const out_buffer_index_offset,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
        SIXTRL_RESTRICT ptr_success_flag );

/* ========================================================================= */

__kernel void NS(Track_particles_line_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_UINT64_T const particle_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_UINT64_T const line_begin_idx, SIXTRL_UINT64_T const line_end_idx,
    SIXTRL_UINT64_T const finish_turn_value,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
        SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)           buf_size_t;
    typedef NS(particle_num_elements_t) num_element_t;

    SIXTRL_STATIC_VAR NS(opencl_success_flag_t) const
        ZERO_FLAG = ( NS(opencl_success_flag_t) )0u;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  be_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*        pb_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     particles_t;

    NS(opencl_success_flag_t) success_flag = ( NS(opencl_success_flag_t) )-1;

    buf_size_t const slot_size  = ( buf_size_t )8u;
    num_element_t particle_id   = ( num_element_t )get_global_id( 0 );
    num_element_t const stride  = ( num_element_t )get_global_size( 0 );
    num_element_t num_particles = ( num_element_t )0u;

    be_iter_t line_begin   = SIXTRL_NULLPTR;
    be_iter_t line_end     = SIXTRL_NULLPTR;
    particles_t particles  = SIXTRL_NULLPTR;

    if( ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) &&
        ( line_begin_idx <= line_end_idx ) &&
        ( line_end_idx <= NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) ) )
    {
        line_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

        if( line_begin != SIXTRL_NULLPTR )
        {
            line_end   = line_begin + line_end_idx;
            line_begin = line_begin + line_begin_idx;

            if( !NS(BeamElements_objects_range_are_all_beam_elements)(
                    line_begin, line_end ) )
            {
                success_flag = ( NS(opencl_success_flag_t) )-8;
                line_begin = line_end = SIXTRL_NULLPTR;
            }
        }
        else
        {
            success_flag = ( NS(opencl_success_flag_t) )-4;
            line_begin = SIXTRL_NULLPTR;
        }
    }
    else
    {
        success_flag = ( NS(opencl_success_flag_t) )-2;
    }

    SIXTRL_ASSERT( ( success_flag != ZERO_FLAG ) ||
        ( ( line_begin == SIXTRL_NULLPTR ) &&
          ( line_end == SIXTRL_NULLPTR ) ) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    if( ( success_flag == ZERO_FLAG ) &&
        ( !NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) ) &&
        ( ( SIXTRL_UINT64_T )NS(ManagedBuffer_get_num_objects)(
            pbuffer, slot_size ) > particle_set_index ) )
    {
        pb_iter_t particles_it = NS(ManagedBuffer_get_objects_index_begin)(
            pbuffer, slot_size );

        if( particles_it != SIXTRL_NULLPTR )
        {
            particles_it = particles_it + particle_set_index;

            if( ( NS(Object_get_type_id)( particles_it ) ==
                  NS(OBJECT_TYPE_PARTICLE) ) &&
                ( NS(Object_get_begin_addr)( particles_it ) !=
                  ( NS(buffer_addr_t) )0u ) )
            {
                particles = ( particles_t )( uintptr_t
                    )NS(Object_get_begin_addr)( particles_it );
            }
            else
            {
                success_flag = ( NS(opencl_success_flag_t) )-64;
            }
        }
        else
        {
            success_flag = ( NS(opencl_success_flag_t) )-32;
        }
    }
    else if( success_flag == ZERO_FLAG )
    {
        success_flag = ( NS(opencl_success_flag_t) )-16;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    if( success_flag == ZERO_FLAG )
    {
        bool const fin = ( bool )( finish_turn_value > 0u );

        for( ; particle_id < num_particles ; particle_id += stride )
        {
            if( success_flag != ZERO_FLAG ) break;

            success_flag = ( ( NS(Track_particle_line)( particles,
                particle_id, line_begin, line_end, fin ) ) == 0 )
                    ? 0 : ( NS(opencl_success_flag_t) )-128;
        }
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    NS(OpenCl1x_collect_success_flag_value)( ptr_success_flag, success_flag );
    return;
}

__kernel void NS(Track_particles_single_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const increment_turn,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
        SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_index_t)                             index_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    NS(opencl_success_flag_t) success_flag = ( NS(opencl_success_flag_t) )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t idx = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_global_size( 0 );

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buffer, slot_size );

        obj_const_iter_t be_end =
            NS(ManagedBuffer_get_const_objects_index_end)(
                belem_buffer, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            particles_buffer, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

        success_flag = ( NS(opencl_success_flag_t) )0u;

        for( ; idx < num_particles ; idx += stride )
        {
            index_t const start_beam_element_id =
                NS(Particles_get_at_element_id_value)( particles, idx );

            success_flag |= NS(Track_particle_beam_elements_obj)(
                particles, idx, be_begin, be_end );

            if( ( success_flag   == ( NS(opencl_success_flag_t) )0 ) &&
                ( increment_turn != ( SIXTRL_INT64_T )0 ) &&
                ( NS(Particles_is_not_lost_value)( particles, idx ) ) )
            {
                NS(Particles_increment_at_turn_value)( particles, idx );
                NS(Particles_set_at_element_id_value)(
                    particles, idx, start_beam_element_id );
            }
        }
    }
    else if( NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) )
    {
        success_flag |= ( NS(opencl_success_flag_t) )-2;
    }
    else if( NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) != ( buf_size_t )1u )
    {
        success_flag |= ( NS(opencl_success_flag_t) )-4;
    }
    else if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
    {
        success_flag |= ( NS(opencl_success_flag_t) )-8;
    }

    NS(OpenCl1x_collect_success_flag_value)( ptr_success_flag, success_flag );
    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_until_turn_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const until_turn,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)*
        SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const  slot_size = ( buf_size_t )8u;
    NS(opencl_success_flag_t) success_flag = ( NS(opencl_success_flag_t) )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) == ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t idx = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_global_size( 0 );

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buffer, slot_size );

        obj_const_iter_t be_end =
            NS(ManagedBuffer_get_const_objects_index_end)(
                belem_buffer, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            particles_buffer, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
            belem_buffer, slot_size ) > ( buf_size_t )0u );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );

        success_flag = ( NS(opencl_success_flag_t) )0u;

        for( ; idx < num_particles ; idx += stride )
        {
            success_flag |= NS(Track_particle_until_turn_obj)(
                particles, idx, be_begin, be_end, until_turn );
        }
    }
    else if( NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) )
    {
        success_flag |= ( NS(opencl_success_flag_t) )-2;
    }
    else if( NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) != ( buf_size_t )1u )
    {
        success_flag |= ( NS(opencl_success_flag_t) )-4;
    }
    else if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
    {
        success_flag |= ( NS(opencl_success_flag_t) )-8;
    }

    NS(OpenCl1x_collect_success_flag_value)( ptr_success_flag, success_flag );

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_elem_by_elem_debug_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config, SIXTRL_INT64_T const until_turn,
    SIXTRL_UINT64_T const out_buffer_index_offset,
    SIXTRL_BUFFER_DATAPTR_DEC NS(opencl_success_flag_t)* SIXTRL_RESTRICT ptr_success_flag )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_index_t)                             index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*  obj_const_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC     NS(Particles)*     ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    NS(opencl_success_flag_t) success = ( NS(opencl_success_flag_t) )-1;

    if( ( !NS(ManagedBuffer_needs_remapping)( elem_by_elem_buffer,
                                              slot_size ) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                elem_by_elem_buffer, slot_size ) > out_buffer_index_offset ) &&
        ( !NS(ManagedBuffer_needs_remapping)( particles_buffer,
                                              slot_size) ) &&
        (  NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) >= ( buf_size_t )1u ) &&
        ( !NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) ) )
    {
        num_element_t idx = ( num_element_t )get_global_id( 0 );
        num_element_t const stride = ( num_element_t )get_global_size( 0 );
        buf_size_t const slot_size = ( buf_size_t )8u;

        obj_const_iter_t be_begin =
            NS(ManagedBuffer_get_const_objects_index_begin)(
                belem_buffer, slot_size );

        obj_const_iter_t be_end =
            NS(ManagedBuffer_get_const_objects_index_end)(
                belem_buffer, slot_size );

        ptr_particles_t particles = NS(Particles_managed_buffer_get_particles)(
            particles_buffer, 0u, slot_size );

        num_element_t const num_particles =
            NS(Particles_get_num_of_particles)( particles );

        NS(ElemByElemConfig_assign_managed_output_buffer)( elem_by_elem_config,
            elem_by_elem_buffer, out_buffer_index_offset, slot_size );

        SIXTRL_ASSERT( particles != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_begin  != SIXTRL_NULLPTR );
        SIXTRL_ASSERT( be_end    != SIXTRL_NULLPTR );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            particles_buffer, slot_size ) );

        SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
            elem_by_elem_buffer, slot_size ) );

        success = ( NS(opencl_success_flag_t) )0u;

        for( ; idx < num_particles ; idx += stride )
        {
            success |= NS(Track_particle_element_by_element_until_turn_objs)(
                particles, idx, elem_by_elem_config,
                    be_begin, be_end, until_turn );
        }
    }
    else if( NS(ManagedBuffer_needs_remapping)( particles_buffer, slot_size) )
    {
        success |= ( NS(opencl_success_flag_t) )-2;
    }
    else if( NS(ManagedBuffer_get_num_objects)(
                particles_buffer, slot_size ) != ( buf_size_t )1u )
    {
        success |= ( NS(opencl_success_flag_t) )-4;
    }
    else if( NS(ManagedBuffer_needs_remapping)( belem_buffer, slot_size ) )
    {
        success |= ( NS(opencl_success_flag_t) )-8;
    }
    else if( NS(ManagedBuffer_needs_remapping)(
            elem_by_elem_buffer, slot_size ) )
    {
        success |= ( NS(opencl_success_flag_t) )-16;
    }

    NS(OpenCl1x_collect_success_flag_value)( ptr_success_flag, success );
    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_DEBUG_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles_debug.cl */
