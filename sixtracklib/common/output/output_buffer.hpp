#ifndef SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_CXX_HPP__
#define SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_CXX_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <vector>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/output/output_buffer.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/particles.hpp"
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    using output_buffer_flag_t = ::NS(output_buffer_flag_t);

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST output_buffer_flag_t
        NS(OUTPUT_BUFFER_NONE) = static_cast< output_buffer_flag_t >(
            SIXTRL_OUTPUT_BUFFER_NONE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST output_buffer_flag_t
        NS(OUTPUT_BUFFER_ELEM_BY_ELEM) = static_cast< output_buffer_flag_t >(
            SIXTRL_OUTPUT_BUFFER_ELEM_BY_ELEM );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST output_buffer_flag_t
        NS(OUTPUT_BUFFER_BEAM_MONITORS) = static_cast< output_buffer_flag_t >(
            SIXTRL_OUTPUT_BUFFER_BEAM_MONITORS );

    /* --------------------------------------------------------------------- */

    SIXTRL_FN bool OutputBuffer_requires_beam_monitor_output(
        output_buffer_flag_t const flags ) SIXTRL_NOEXCEPT;

    SIXTRL_FN bool OutputBuffer_requires_elem_by_elem_output(
        output_buffer_flag_t const flags ) SIXTRL_NOEXCEPT;

    SIXTRL_FN bool OutputBuffer_requires_output_buffer(
        output_buffer_flag_t const flags ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN output_buffer_flag_t OutputBuffer_required_for_tracking(
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer::size_type const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN output_buffer_flag_t OutputBuffer_required_for_tracking(
        const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
        const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN int OutputBuffer_prepare(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer& SIXTRL_RESTRICT_REF output_buffer,
        Particles const& SIXTRL_RESTRICT_REF particles,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        Buffer::size_type* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        Particles::index_t* SIXTRL_RESTRICT ptr_min_turn_id );

    SIXTRL_HOST_FN int OutputBuffer_prepare(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer& SIXTRL_RESTRICT_REF output_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        Buffer::size_type* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id );

    template< typename Iter >
    SIXTRL_HOST_FN int OutputBuffer_prepare(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer& SIXTRL_RESTRICT_REF output_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particle_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        Buffer::size_type* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_HOST_FN int OutputBuffer_prepare(
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        const ::NS(Particles) *const SIXTRL_RESTRICT particles,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id );

    SIXTRL_HOST_FN int OutputBuffer_prepare(
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id );

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN int OutputBuffer_calculate_output_buffer_params(
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Particles const& SIXTRL_RESTRICT_REF particles,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_objects,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_slots,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_data_ptrs,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_garbage,
        Buffer::size_type const slot_size =
            ::NS(BUFFER_DEFAULT_SLOT_SIZE) ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN int OutputBuffer_calculate_output_buffer_params(
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_objects,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_slots,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_data_ptrs,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_garbage,
        Buffer::size_type const slot_size =
            ::NS(BUFFER_DEFAULT_SLOT_SIZE) ) SIXTRL_NOEXCEPT;

    template< typename Iter >
    SIXTRL_HOST_FN int OutputBuffer_calculate_output_buffer_params(
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_objects,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_slots,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_data_ptrs,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_garbage,
        Buffer::size_type const slot_size = ::NS(BUFFER_DEFAULT_SLOT_SIZE) );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_HOST_FN int OutputBuffer_calculate_output_buffer_params(
        const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        const ::NS(Particles) *const SIXTRL_RESTRICT particles,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
        ::NS(buffer_size_t) const slot_size =
            ::NS(BUFFER_DEFAULT_SLOT_SIZE) ) SIXTRL_NOEXCEPT;

    SIXTRL_HOST_FN int OutputBuffer_calculate_output_buffer_params(
        const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
        ::NS(buffer_size_t) const slot_size =
            ::NS(BUFFER_DEFAULT_SLOT_SIZE) ) SIXTRL_NOEXCEPT;
}

#endif /* defined(  __cplusplus ) && !defined( _GPUCODE ) */

/* ========================================================================= */
/* ========              Inline function implementation            ========= */
/* ========================================================================= */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE bool OutputBuffer_requires_beam_monitor_output(
        output_buffer_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        return ::NS(OutputBuffer_requires_beam_monitor_output)( flags );
    }

    SIXTRL_INLINE bool OutputBuffer_requires_elem_by_elem_output(
        output_buffer_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        return ::NS(OutputBuffer_requires_elem_by_elem_output)( flags );
    }

    SIXTRL_INLINE bool OutputBuffer_requires_output_buffer(
        output_buffer_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        return ::NS(OutputBuffer_requires_output_buffer)( flags );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE output_buffer_flag_t OutputBuffer_required_for_tracking(
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer::size_type const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT
    {
        return ::NS(OutputBuffer_required_for_tracking)(
            particles_buffer.getCApiPtr(), beam_elements_buffer.getCApiPtr(),
                dump_elem_by_elem_turns );
    }

    SIXTRL_INLINE output_buffer_flag_t OutputBuffer_required_for_tracking(
        const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
        const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT
    {
        return ::NS(OutputBuffer_required_for_tracking)(
            particles_buffer, beam_elements_buffer, dump_elem_by_elem_turns );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE int OutputBuffer_prepare(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer& SIXTRL_RESTRICT_REF output_buffer,
        Particles const& SIXTRL_RESTRICT_REF particles,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        Buffer::size_type* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        Particles::index_t* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        return ::NS(OutputBuffer_prepare)( beam_elements_buffer.getCApiPtr(),
            output_buffer.getCApiPtr(), particles.getCApiPtr(),
                dump_elem_by_elem_turns, ptr_elem_by_elem_out_index_offset,
                    ptr_beam_monitor_out_index_offset, ptr_min_turn_id );
    }

    SIXTRL_INLINE int OutputBuffer_prepare(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer& SIXTRL_RESTRICT_REF output_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        Buffer::size_type* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        return ::NS(OutputBuffer_prepare_for_particle_sets)(
            beam_elements_buffer.getCApiPtr(), output_buffer.getCApiPtr(),
                particles_buffer.getCApiPtr(), num_particle_sets,
                    particle_set_indices_begin, dump_elem_by_elem_turns,
                        ptr_elem_by_elem_out_index_offset,
                        ptr_beam_monitor_out_index_offset, ptr_min_turn_id );
    }

    template< typename Iter >
    SIXTRL_INLINE int OutputBuffer_prepare(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer& SIXTRL_RESTRICT_REF output_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particle_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        Buffer::size_type* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        std::vector< ::NS(buffer_size_t) > temp_store(
            particle_set_indices_begin, particle_set_indices_end );

        return ::NS(OutputBuffer_prepare_for_particle_sets)(
            beam_elements_buffer.getCApiPtr(), output_buffer.getCApiPtr(),
                particle_buffer.getCApiPtr(), temp_store.size(),
                    temp_store.data(), dump_elem_by_elem_turns,
                        ptr_elem_by_elem_out_index_offset,
                        ptr_beam_monitor_out_index_offset, ptr_min_turn_id );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE int OutputBuffer_prepare(
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        const ::NS(Particles) *const SIXTRL_RESTRICT particles,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        return ::NS(OutputBuffer_prepare)( beam_elements_buffer, output_buffer,
            particles, dump_elem_by_elem_turns,
                ptr_elem_by_elem_out_index_offset,
                    ptr_beam_monitor_out_index_offset, ptr_min_turn_id );
    }

    SIXTRL_INLINE int OutputBuffer_prepare(
        ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        const ::NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_elem_by_elem_out_index_offset,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_beam_monitor_out_index_offset,
        ::NS(particle_index_t)* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        return ::NS(OutputBuffer_prepare_for_particle_sets)(
            beam_elements_buffer, output_buffer, particles_buffer,
                num_particle_sets, particle_set_indices_begin,
                    dump_elem_by_elem_turns, ptr_elem_by_elem_out_index_offset,
                        ptr_beam_monitor_out_index_offset, ptr_min_turn_id );
    }

    /* --------------------------------------------------------------------- */

//     SIXTRL_INLINE int OutputBuffer_calculate_output_buffer_params(
//         Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
//         Particles const& SIXTRL_RESTRICT_REF particles,
//         Buffer::size_type const dump_elem_by_elem_turns,
//         Buffer::size_type* SIXTRL_RESTRICT ptr_num_objects,
//         Buffer::size_type* SIXTRL_RESTRICT ptr_num_slots,
//         Buffer::size_type* SIXTRL_RESTRICT ptr_num_data_ptrs,
//         Buffer::size_type* SIXTRL_RESTRICT ptr_num_garbage,
//         Buffer::size_type const slot_size ) SIXTRL_NOEXCEPT
//     {
//         return ::NS(OutputBuffer_calculate_output_buffer_params)(
//             beam_elements_buffer.getCApiPtr(), particles.getCApiPtr(),
//                 dump_elem_by_elem_turns, ptr_num_objects, ptr_num_slots,
//                     ptr_num_data_ptrs, ptr_num_garbage, slot_size );
//     }

    SIXTRL_INLINE int OutputBuffer_calculate_output_buffer_params(
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_particle_sets,
        Buffer::size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_objects,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_slots,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_data_ptrs,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_garbage,
        Buffer::size_type const slot_size ) SIXTRL_NOEXCEPT
    {
        return
        ::NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
            beam_elements_buffer.getCApiPtr(), particles_buffer.getCApiPtr(),
                num_particle_sets, particle_set_indices_begin,
                    dump_elem_by_elem_turns, ptr_num_objects, ptr_num_slots,
                        ptr_num_data_ptrs, ptr_num_garbage, slot_size );
    }

    template< typename Iter >
    SIXTRL_INLINE int OutputBuffer_calculate_output_buffer_params(
        Buffer const& SIXTRL_RESTRICT_REF beam_elements_buffer,
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Iter particle_set_indices_begin, Iter particle_set_indices_end,
        Buffer::size_type const dump_elem_by_elem_turns,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_objects,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_slots,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_data_ptrs,
        Buffer::size_type* SIXTRL_RESTRICT ptr_num_garbage,
        Buffer::size_type const slot_size )
    {
        std::vector< ::NS(buffer_size_t) > temp_store(
            particle_set_indices_begin, particle_set_indices_end );

        return
        ::NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
            beam_elements_buffer.getCApiPtr(), particles_buffer.getCApiPtr(),
                temp_store.size(), temp_store.data(), dump_elem_by_elem_turns,
                    ptr_num_objects, ptr_num_slots, ptr_num_data_ptrs,
                        ptr_num_garbage, slot_size );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE int OutputBuffer_calculate_output_buffer_params(
        const ::NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        const ::NS(Particles) *const SIXTRL_RESTRICT particles,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
        ::NS(buffer_size_t) const slot_size  ) SIXTRL_NOEXCEPT
    {
        return
        ::NS(OutputBuffer_calculate_output_buffer_params)(
            beam_elements_buffer, particles, dump_elem_by_elem_turns,
                ptr_num_objects, ptr_num_slots, ptr_num_data_ptrs,
                    ptr_num_garbage, slot_size );
    }

    SIXTRL_INLINE int OutputBuffer_calculate_output_buffer_params(
        const NS(Buffer) *const SIXTRL_RESTRICT beam_elements_buffer,
        const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_particle_sets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
        ::NS(buffer_size_t) const dump_elem_by_elem_turns,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_objects,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_slots,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_data_ptrs,
        ::NS(buffer_size_t)* SIXTRL_RESTRICT ptr_num_garbage,
        ::NS(buffer_size_t) const slot_size  ) SIXTRL_NOEXCEPT
    {
        return
        ::NS(OutputBuffer_calculate_output_buffer_params_for_particles_sets)(
            beam_elements_buffer, particles_buffer, num_particle_sets,
                particle_set_indices_begin, dump_elem_by_elem_turns,
                    ptr_num_objects, ptr_num_slots, ptr_num_data_ptrs,
                        ptr_num_garbage, slot_size );
    }
}

#endif /* !defined( __cplusplus ) && !defined( _GPUCODE ) */


#endif /* SIXTRACKL_COMMON_OUTPUT_OUTPUT_BUFFER_CXX_HPP__ */

/* end: sixtracklib/common/output/output_buffer.hpp */
