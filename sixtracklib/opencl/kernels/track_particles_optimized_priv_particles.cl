#ifndef SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__
#define SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/default_compile_options.h"
    #include "sixtracklib/opencl/internal/success_flag.h"
    #include "sixtracklib/opencl/internal/optimized_priv_particle.h"

    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/managed_buffer_minimal.h"
    #include "sixtracklib/common/buffer/managed_buffer_remap.h"
    #include "sixtracklib/common/internal/buffer_object_defines.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

__kernel void NS(Track_particles_line_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_UINT64_T const particle_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_UINT64_T const line_begin_idx, SIXTRL_UINT64_T const line_end_idx,
    SIXTRL_UINT64_T const finish_turn_value );

__kernel void NS(Track_particles_single_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const increment_turn );

__kernel void NS(Track_particles_until_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_INT64_T const until_turn );

__kernel void NS(Track_particles_elem_by_elem_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config,
    SIXTRL_INT64_T const until_turn,
    SIXTRL_UINT64_T const out_buffer_index_offset );

/* ========================================================================= */

__kernel void NS(Track_particles_line_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT pbuffer,
    SIXTRL_UINT64_T const particle_set_index,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const* SIXTRL_RESTRICT belem_buffer,
    SIXTRL_UINT64_T const line_begin_idx, SIXTRL_UINT64_T const line_end_idx,
    SIXTRL_UINT64_T const finish_turn_value )
{
    typedef NS(buffer_size_t)           buf_size_t;
    typedef NS(particle_num_elements_t) num_element_t;
    typedef NS(particle_real_t)         real_t;
    typedef NS(particle_index_t)        index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_iter_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*       pb_iter_t;
    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)* ptr_particles_t;

    buf_size_t const slot_size  = ( buf_size_t )8u;
    num_element_t particle_id   = ( num_element_t )get_global_id( 0 );
    num_element_t const stride  = ( num_element_t )get_global_size( 0 );
    num_element_t num_particles = ( num_element_t )0u;

    bool const fin = ( bool )( finish_turn_value > ( SIXTRL_UINT64_T )0u );

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    real_t reals[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    index_t indexes[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    ptr_particles_t in_particles = SIXTRL_NULLPTR;
    pb_iter_t part_it            = SIXTRL_NULLPTR;
    be_iter_t line_begin         = SIXTRL_NULLPTR;
    be_iter_t line_end           = SIXTRL_NULLPTR;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        belem_buffer, slot_size ) );

    line_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buffer, slot_size );

    SIXTRL_ASSERT( line_begin != SIXSTRL_NULLPTR );
    SIXTRL_ASSERT( line_begin_idx <= line_end_idx );
    SIXTRL_ASSERT( line_end_idx <= NS(ManagedBuffer_get_num_objects)(
        belem_buffer, slot_size ) );

    line_end   = line_begin + line_end_idx;
    line_begin = line_begin + line_begin_idx;

    SIXTRL_ASSERT( NS(BeamElements_objects_range_are_all_beam_elements)(
        line_begin, line_end ) );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)( pbuffer, slot_size ) );

    part_it = NS(ManagedBuffer_get_objects_index_begin)( pbuffer, slot_size );
    SIXTRL_ASSERT( part_it != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( ( SIXTRL_UINT64_T )NS(ManagedBuffer_get_num_objects)(
        pbuffer, slot_size ) > particle_set_index );

    part_it = part_it + particle_set_index;

    SIXTRL_ASSERT( NS(Object_get_type_id)( part_it ) ==
                   NS(OBJECT_TYPE_PARTICLE) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( part_it ) >
                   NS(buffer_addr_t)0u );

    in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( part_it );

    SIXTRL_ASSERT( in_particles != SIXTRL_NULLPTR );
    num_particles = in_particles->num_particles;

    NS(OpenCl1x_init_optimized_priv_particle)(
        &particles, &reals[ 0 ], &indexes[ 0 ] );

    SIXTRL_ASSERT( particles.num_particles == ( num_element_t )1u );

    for( ; particle_id < num_particles ; particle_id += stride )
    {
        NS(Particles_copy_from_generic_addr_data)(
            &particles, 0, in_particles, particle_id );

        NS(Track_particle_line)( &particles, 0, line_begin, line_end, fin );

        NS(Particles_copy_to_generic_addr_data)(
            in_particles, particle_id, &particles, 0 );
    }

    return;
}

__kernel void NS(Track_particles_single_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const increment_turn )
{
    typedef NS(buffer_size_t)                                   buf_size_t;
    typedef NS(particle_num_elements_t)                         num_element_t;
    typedef NS(particle_real_t)                                 real_t;
    typedef NS(particle_index_t)                                index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*
            obj_iter_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*
            obj_const_iter_t;

    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)*
            ptr_particles_t;

    num_element_t idx = ( num_element_t )get_global_id( 0 );
    buf_size_t const slot_size  = ( buf_size_t )8u;
    num_element_t const stride  = ( num_element_t )get_global_size( 0 );
    num_element_t num_particles = ( num_element_t )0u;

    obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buffer, slot_size );

    obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buffer, slot_size );

    obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
        particles_buffer, slot_size );

    ptr_particles_t in_particles = SIXTRL_NULLPTR;

    real_t reals[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    index_t indexes[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    NS(OpenCl1x_init_optimized_priv_particle)(
        &particles, &reals[ 0 ], &indexes[ 0 ] );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( pb_it    != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                   NS(OBJECT_TYPE_PARTICLE) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                   ( NS(buffer_addr_t) )0u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size ) >= ( buf_size_t )1u );

    in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pb_it );

    SIXTRL_ASSERT( ( in_particles == SIXTRL_NULLPTR ) ||
           ( ( in_particles->particle_id   != SIXTRL_NULLPTR ) &&
             ( in_particles->at_element_id != SIXTRL_NULLPTR ) ) );

    num_particles = ( in_particles != SIXTRL_NULLPTR )
        ? in_particles->num_particles : ( num_element_t )0u;

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        particles_buffer, slot_size ) );

    SIXTRL_ASSERT( !NS(ManagedBuffer_needs_remapping)(
        belem_buffer, slot_size ) );

    for( ; idx < num_particles ; idx += stride )
    {
        index_t start_element_id = ( index_t )0u;

        NS(Particles_copy_from_generic_addr_data)(
            &particles, 0, in_particles, idx );

        start_element_id = NS(Particles_get_at_element_id_value)(
            &particles, 0 );

        NS(Track_particle_beam_elements_obj)(
            &particles, 0, be_begin, be_end );

        if( ( increment_turn != ( SIXTRL_INT64_T )0 ) &&
            ( NS(Particles_is_not_lost_value)( &particles, 0 ) ) )
        {
            NS(Track_particle_increment_at_turn)(
                &particles, 0, start_element_id );
        }

        NS(Particles_copy_to_generic_addr_data)(
            in_particles, idx, &particles, 0 );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_until_turn_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char* SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer, SIXTRL_INT64_T const until_turn )
{
    typedef NS(buffer_size_t)                                buf_size_t;
    typedef NS(particle_num_elements_t)                      num_element_t;
    typedef NS(particle_real_t)                              real_t;
    typedef NS(particle_index_t)                             index_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)*
            obj_iter_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*
            obj_const_iter_t;

    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)*
            ptr_particles_t;

    buf_size_t const slot_size = ( buf_size_t )8u;
    num_element_t const stride = ( num_element_t )get_global_size( 0 );
    num_element_t num_particles = ( num_element_t )0u;
    num_element_t idx = ( num_element_t )get_global_id( 0 );

    obj_const_iter_t be_begin = NS(ManagedBuffer_get_const_objects_index_begin)(
        belem_buffer, slot_size );

    obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buffer, slot_size );

    obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
        particles_buffer, slot_size );

    ptr_particles_t in_particles = SIXTRL_NULLPTR;

    real_t reals[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    index_t indexes[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;
    NS(OpenCl1x_init_optimized_priv_particle)(
        &particles, &reals[ 0 ], &indexes[ 0 ] );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( pb_it    != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                   NS(OBJECT_TYPE_PARTICLE) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                  ( NS(buffer_addr_t) )0u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size ) >= ( buf_size_t )1u );

    in_particles = ( ptr_particles_t )(
        uintptr_t )NS(Object_get_begin_addr)( pb_it );

    num_particles = ( in_particles != SIXTRL_NULLPTR )
        ? in_particles->num_particles : ( num_element_t )0u;

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        belem_buffer, slot_size ) > ( buf_size_t )0u );

    SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
        particles_buffer, slot_size ) );

    for( ; idx < num_particles ; idx += stride )
    {
        NS(Particles_copy_from_generic_addr_data)(
            &particles, 0, in_particles, idx );

        NS(Track_particle_until_turn_obj)(
            &particles, 0, be_begin, be_end, until_turn );

        NS(Particles_copy_to_generic_addr_data)(
            in_particles, idx, &particles, 0 );
    }

    return;
}

/* ------------------------------------------------------------------------- */

__kernel void NS(Track_particles_elem_by_elem_opt_pp_opencl)(
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT particles_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
        SIXTRL_RESTRICT belem_buffer,
    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
        SIXTRL_RESTRICT elem_by_elem_buffer,
    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC NS(ElemByElemConfig)*
        SIXTRL_RESTRICT elem_by_elem_config, SIXTRL_INT64_T const until_turn,
    SIXTRL_UINT64_T const out_buffer_index_offset )
{
    typedef NS(buffer_size_t)                         buf_size_t;
    typedef NS(particle_num_elements_t)               num_element_t;
    typedef NS(particle_real_t)                       real_t;
    typedef NS(particle_index_t)                      index_t;
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object)* obj_iter_t;

    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC  NS(Object) const*
            obj_const_iter_t;

    typedef SIXTRL_BUFFER_DATAPTR_DEC NS(ParticlesGenericAddr)*
            ptr_particles_t;

    SIXTRL_STATIC_VAR NS(opencl_success_flag_t) const ZERO_FLAG =
        ( NS(opencl_success_flag_t) )0u;

    num_element_t idx = ( num_element_t )get_global_id( 0 );
    num_element_t num_particles = ( num_element_t )0u;
    num_element_t const stride  = ( num_element_t )get_global_size( 0 );
    buf_size_t const slot_size  = ( buf_size_t )8u;

    obj_const_iter_t be_begin =
        NS(ManagedBuffer_get_const_objects_index_begin)(
            belem_buffer, slot_size );

    obj_const_iter_t be_end = NS(ManagedBuffer_get_const_objects_index_end)(
        belem_buffer, slot_size );

    obj_iter_t pb_it = NS(ManagedBuffer_get_objects_index_begin)(
        particles_buffer, slot_size );

    obj_iter_t elem_by_elem_it = NS(ManagedBuffer_get_objects_index_begin)(
        elem_by_elem_buffer, slot_size );

    ptr_particles_t in_particles           = SIXTRL_NULLPTR;
    ptr_particles_t elem_by_elem_particles = SIXTRL_NULLPTR;

    real_t reals[] =
    {
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0,
        ( real_t )0.0, ( real_t )0.0, ( real_t )0.0
    };

    index_t indexes[] =
    {
        ( index_t )0, ( index_t )0, ( index_t )0, ( index_t )0
    };

    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles) particles;

    NS(OpenCl1x_init_optimized_priv_particle)(
        &particles, &reals[ 0 ], &indexes[ 0 ] );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( be_end   != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( pb_it    != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(Object_get_type_id)( pb_it ) !=
                   NS(OBJECT_TYPE_PARTICLE) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( pb_it ) !=
                   ( NS(buffer_addr_t) )0u );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        particles_buffer, slot_size ) >= ( buf_size_t )1u );

    in_particles = ( ptr_particles_t )( uintptr_t
        )NS(Object_get_begin_addr)( pb_it );

    SIXTRL_ASSERT( ( in_particles == SIXTRL_NULLPTR ) ||
           ( ( in_particles->particle_id   != SIXTRL_NULLPTR ) &&
             ( in_particles->at_element_id != SIXTRL_NULLPTR ) &&
             ( in_particles->at_turn       != SIXTRL_NULLPTR ) ) );

    num_particles = ( in_particles != SIXTRL_NULLPTR )
            ? in_particles->num_particles : ( num_element_t )0u;

    SIXTRL_ASSERT( elem_by_elem_it != SIXTRL_NULLPTR );

    SIXTRL_ASSERT( NS(ManagedBuffer_get_num_objects)(
        elem_by_elem_buffer, slot_size ) > out_buffer_index_offset );

    SIXTRL_ASSERT( NS(Particles_managed_buffer_is_particles_buffer)(
        elem_by_elem_buffer, slot_size ) );

    elem_by_elem_it = elem_by_elem_it + out_buffer_index_offset;

    SIXTRL_ASSERT( NS(Object_get_type_id)( elem_by_elem_it ) ==
                   NS(OBJECT_TYPE_PARTICLE) );

    SIXTRL_ASSERT( NS(Object_get_begin_addr)( elem_by_elem_it ) !=
                   ( NS(buffer_addr_t) )0u );

    NS(ElemByElemConfig_set_output_store_address)( elem_by_elem_config,
        ( uintptr_t )NS(Object_get_begin_addr)( elem_by_elem_it ) );

    elem_by_elem_particles = ( ptr_particles_t )( uintptr_t
       )NS(Object_get_begin_addr)( elem_by_elem_it );

    SIXTRL_ASSERT(
       ( elem_by_elem_particles == SIXTRL_NULLPTR ) ||
       ( elem_by_elem_particles->num_particles >=
           NS(ElemByElemConfig_get_num_particles_to_store)(
               elem_by_elem_config ) ) );

    for( ; idx < num_particles ; idx += stride )
    {
        index_t start_elem_id = ( index_t )0u;
        index_t particle_id   = ( index_t )0u;
        index_t turn          = ( index_t )0u;

        NS(opencl_success_flag_t) success =
            NS(Particles_copy_from_generic_addr_data)(
                &particles, 0, in_particles, idx );

        start_elem_id = NS(Particles_get_at_element_id_value)( &particles, 0 );
        particle_id = NS(Particles_get_particle_id_value)( &particles, 0 );
        turn = NS(Particles_get_at_turn_value)( &particles, 0 );

        while( ( success == ZERO_FLAG ) && ( turn < until_turn ) )
        {
            obj_const_iter_t be_it = be_begin;
            index_t at_elem_id = start_elem_id;

            while( ( success == ZERO_FLAG ) && ( be_it != be_end ) )
            {
                num_element_t const store_idx =
                    NS(ElemByElemConfig_get_particles_store_index_details)(
                        elem_by_elem_config,
                            particle_id, at_elem_id++, turn );

                success = NS(Particles_copy_to_generic_addr_data)(
                    elem_by_elem_particles, store_idx, &particles, 0 );

                if( success == ZERO_FLAG )
                {
                    success = NS(Track_particle_beam_element_obj)(
                        &particles, 0, be_it++ );
                }
            }

            if( success == ZERO_FLAG )
            {
                SIXTRL_ASSERT( NS(Particles_is_not_lost_value)(
                    &particles, 0 ) );

                NS(Track_particle_increment_at_turn)(
                    &particles, 0, start_elem_id );

                turn = NS(Particles_get_at_turn_value)( &particles, 0 );
            }
        }

        if( success != ZERO_FLAG )
        {
            SIXTRL_ASSERT( NS(Particles_is_not_lost_value)(
                    &particles, 0 ) );

            success = NS(Particles_copy_to_generic_addr_data)(
                in_particles, idx, &particles, 0 );
        }

        if( success != ZERO_FLAG ) break;
    }

    return;
}

#endif /* SIXTRACKLIB_OPENCL_INTERNAL_TRACK_PARTICLES_OPTIMIZED_PRIV_PARTICLES_KERNEL_CL__ */

/* end: sixtracklib/opencl/kernels/track_particles_optimized_priv_particles.cl */
