#include "sixtracklib/common/internal/track_job_base.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
        #include <string>
        #include <vector>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
        #include <limits.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/context/context_abs_base.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    char const* TrackJobBase::configString() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str.c_str();
    }

    /* ---------------------------------------------------------------- */

    bool TrackJobBase::hasContext() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_ptr_context.get() != nullptr ) &&
                 ( this->m_ptr_context->type() == CONTEXT_TYPE_INVALID ) );
    }

    TrackJobBase::context_t const*
    TrackJobBase::context() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_context.get();
    }

    TrackJobBase::context_t* TrackJobBase::context() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_context.get();
    }

    /* ---------------------------------------------------------------- */

    TrackJobBase::elem_by_elem_config_t const&
    TrackJobBase::elemByElemConfig() const SIXTRL_NOEXCEPT
    {
        return this->m_elem_by_elem_config;
    }

    TrackJobBase::elem_by_elem_config_t const*
    TrackJobBase::ptrElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        return &this->m_elem_by_elem_config;
    }

    TrackJobBase::elem_by_elem_order_t
    TrackJobBase::elemByElemStoreOrder() const SIXTRL_NOEXCEPT
    {
        return this->m_elem_by_elem_order;
    }

    void TrackJobBase::setElemByElemStoreOrder(
        TrackJobBase::elem_by_elem_order_t const order )
    {
        this->m_elem_by_elem_order = order;
        return;
    }

    /* ---------------------------------------------------------------- */

    TrackJobBase::track_status_t
    TrackJobBase::lastTrackStatus() const SIXTRL_NOEXCEPT
    {
        return this->m_last_track_status;
    }

    TrackJobBase::c_buffer_t const*
    TrackJobBase::ptrOutputBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrOutputBuffer();
    }

    TrackJobBase::c_buffer_t* TrackJobBase::ptrOutputBuffer() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrOutputBuffer();
    }

    TrackJobBase::buffer_t const&
    TrackJobBase::outputBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetOutputBuffer();
    }

    TrackJobBase::c_buffer_t const*
    TrackJobBase::ptrParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrParticlesBuffer();
    }

    TrackJobBase::c_buffer_t* TrackJobBase::ptrParticlesBuffer() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrParticlesBuffer();
    }

    TrackJobBase::buffer_t const&
    TrackJobBase::particlesBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetParticlesBuffer();
    }

    TrackJobBase::c_buffer_t const*
    TrackJobBase::ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrBeamElementsBuffer();
    }

    TrackJobBase::c_buffer_t* TrackJobBase::ptrBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        return this->doGetPtrBeamElementsBuffer();
    }

    TrackJobBase::buffer_t const&
    TrackJobBase::beamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetBeamElementsBuffer();
    }

    /* ---------------------------------------------------------------- */

    TrackJobBase::TrackJobBase() SIXTRL_NOEXCEPT :
        m_elem_by_elem_config(),
        m_config_str(),
        m_particle_indices(),
        m_particles_buffer_wrapper(),
        m_beam_elements_buffer_wrapper(),
        m_output_buffer_wrapper(),
        m_ptr_particles_buffer( nullptr ),
        m_ptr_beam_elements_buffer( nullptr ),
        m_ptr_output_buffer( nullptr ),
        m_ptr_context( nullptr ),
        m_last_track_status( TrackJobBase::track_status_t{ -1 } ),
        m_elem_by_elem_order( ::NS(ELEM_BY_ELEM_ORDER_DEFAULT) ),
        m_elem_by_elem_out_buffer_index( TrackJobBase::size_type{ 0 } ),
        m_beam_monitor_out_buffer_index_offset( TrackJobBase::size_type{ 0 } )
    {
        ::NS(Buffer_free)( this->m_particles_buffer_wrapper.getCApiPtr() );
        ::NS(Buffer_free)( this->m_beam_elements_buffer_wrapper.getCApiPtr() );
        ::NS(Buffer_free)( this->m_output_buffer_wrapper.getCApiPtr() );
    }

    TrackJobBase::TrackJobBase( TrackJobBase::ptr_context_t&& context ) :
        m_elem_by_elem_config(),
        m_config_str(),
        m_particle_indices(),
        m_particles_buffer_wrapper(),
        m_beam_elements_buffer_wrapper(),
        m_output_buffer_wrapper(),
        m_ptr_particles_buffer( nullptr ),
        m_ptr_beam_elements_buffer( nullptr ),
        m_ptr_output_buffer( nullptr ),
        m_ptr_context( std::move( context ) ),
        m_last_track_status( TrackJobBase::track_status_t{ -1 } ),
        m_elem_by_elem_order( ::NS(ELEM_BY_ELEM_ORDER_DEFAULT) ),
        m_elem_by_elem_out_buffer_index( TrackJobBase::size_type{ 0 } ),
        m_beam_monitor_out_buffer_index_offset( TrackJobBase::size_type{ 0 } )
    {
        ::NS(Buffer_free)( this->m_particles_buffer_wrapper.getCApiPtr() );
        ::NS(Buffer_free)( this->m_beam_elements_buffer_wrapper.getCApiPtr() );
        ::NS(Buffer_free)( this->m_output_buffer_wrapper.getCApiPtr() );
    }

    bool TrackJobBase::doPerformConfig(
        char const* SIXTRL_RESTRICT config_str )
    {
        return this->doPerformConfigBaseImpl( config_str );
    }

    bool TrackJobBase::doInitOutputBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        const TrackJobBase::c_buffer_t *const SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobBase::size_type const until_turn,
        TrackJobBase::size_type const num_elem_by_elem_turns,
        TrackJobBase::size_type const out_buffer_index_offset )
    {
        return this->doInitOutputBufferBaseImpl(
            output_buffer, particles_buffer, belements_buffer,
            until_turn, num_elem_by_elem_turns, out_buffer_index_offset );
    }

    TrackJobBase::track_status_t TrackJobBase::doTrackUntilTurn(
        TrackJobBase::size_type const until_turn,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        return this->doTrackUntilTurnBaseImpl( until_turn, output_buffer );
    }

    TrackJobBase::track_status_t TrackJobBase::doTrackElemByElem(
        TrackJobBase::size_type const elem_by_elem_turns,
        TrackJobBase::elem_by_elem_config_t const*
            SIXTRL_RESTRICT elem_by_elem_config,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        return this->doTrackElemByElemBaseImpl(
            elem_by_elem_turns, elem_by_elem_config, output_buffer );
    }

    bool TrackJobBase::doCollectParticlesBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particle_buffer )
    {
        return this->doCollectParticlesBufferBaseImpl( particle_buffer );
    }

    bool TrackJobBase::doCollectBeamElementsBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer )
    {
        return this->doCollectBeamElementsBufferBaseImpl(
            beam_elements_buffer );
    }

    bool TrackJobBase::doCollectOutputBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        return this->doCollectOutputBufferBaseImpl( output_buffer );
    }

    bool TrackJobBase::doInitOutputBufferBaseImpl(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        const TrackJobBase::c_buffer_t *const SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobBase::size_type const until_turn,
        TrackJobBase::size_type const num_elem_by_elem_turns,
        TrackJobBase::size_type const out_buffer_index_offset )
    {
        using size_t     = TrackJobBase::size_type;
        using index_t    = ::NS(particle_index_t);
        using buf_size_t = ::NS(buffer_size_t);

        int success = int{ -1 };

        if( ( output_buffer != nullptr ) &&
            ( this->doGetParticleIndexBufferSize() > size_t{ 0 } ) )
        {
            std::vector< NS(buffer_size_t) > const particle_indices(
                this->constParticleIndexBegin(),
                this->constParticleIndexEnd() );

            index_t const min_index_value =
                std::numeric_limits< index_t >::min();

            index_t const max_index_value =
                std::numeric_limits< index_t >::max();

            index_t min_particle_id = max_index_value;
            index_t max_particle_id = min_index_value;

            index_t min_element_id  = max_index_value;
            index_t max_element_id  = min_index_value;

            index_t min_turn_id     = max_index_value;
            index_t max_turn_id     = min_index_value;

            buf_size_t const num_part_indices = particle_indices.size();

            success =
            NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
                particles_buffer, particle_indices.data(), num_part_indices,
                &min_particle_id, &max_particle_id, &min_element_id,
                &max_element_id,  &min_turn_id, &max_turn_id );

            SIXTRL_ASSERT( ( success != 0 ) ||
                           ( ( min_element_id  >= index_t{ 0 } ) &&
                             ( min_element_id  <= max_element_id ) &&
                             ( min_particle_id >= index_t{ 0 } ) &&
                             ( min_particle_id <= max_particle_id ) &&
                             ( min_turn_id     >= index_t{ 0 } ) &&
                             ( min_turn_id     <= max_turn_id  ) ) );

            if( ( success == 0 ) && ( num_elem_by_elem_turns > size_t{ 0 } ) )
            {
                index_t const end_turn_id =
                    min_turn_id + num_elem_by_elem_turns;

                index_t temp_min_element_id = min_element_id;
                index_t temp_max_element_id = max_element_id;

                success =
                NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
                    belements_buffer, &temp_min_element_id,
                        &temp_max_element_id, SIXTRL_NULLPTR, 0 );

                if( success == 0 )
                {
                    if( end_turn_id > max_turn_id + index_t{ 1 } )
                    {
                        max_turn_id = end_turn_id - index_t{ 1 };
                    }

                    if( min_element_id > temp_min_element_id )
                    {
                        min_element_id = temp_min_element_id;
                    }

                    if( max_element_id < temp_max_element_id )
                    {
                        max_element_id = temp_max_element_id;
                    }
                }

                success = NS(ElemByElemConfig_init_detailed)(
                    this->doGetPtrElemByElemConfig(),
                    this->elemByElemStoreOrder(),
                    min_particle_id, max_particle_id, min_element_id,
                    max_element_id, min_turn_id, max_turn_id );

                if( success == 0 )
                {
                    size_t elem_by_elem_out_buffer_index_offset =
                        NS(Buffer_get_num_of_objects)( output_buffer );

                    success =
                        NS(ElemByElemConfig_prepare_output_buffer_from_conf)(
                            this->doGetPtrElemByElemConfig(), output_buffer,
                            &elem_by_elem_out_buffer_index_offset );

                    if( success == 0 )
                    {
                        this->doSetElemByElemOutBufferIndex(
                            elem_by_elem_out_buffer_index_offset );
                    }
                }
            }

            if( ( success == 0 ) &&
                ( until_turn > static_cast< size_t >( min_turn_id ) ) &&
                ( NS(BeamMonitor_are_present_in_buffer)( belements_buffer ) ) )
            {
                buf_size_t beam_monitor_out_buffer_index_offset =
                    NS(Buffer_get_num_of_objects)( output_buffer );

                success = NS(BeamMonitor_prepare_output_buffer_detailed)(
                    belements_buffer, output_buffer,
                    min_particle_id, max_particle_id, min_turn_id,
                    &beam_monitor_out_buffer_index_offset );

                if( success == 0 )
                {
                    this->doSetBeamMonitorOutBufferIndexOffset(
                        beam_monitor_out_buffer_index_offset );
                }
            }
        }

        return success;
    }
}

#else /* defined( __cplusplus ) */

#endif /* defined( __cplusplus ) */

/* end: sixtracklib/common/internal/track_job_base.cpp */

