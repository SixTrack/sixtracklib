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

    bool TrackJobBase::doInitBuffers(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobBase::size_type const num_elem_by_elem_turns,
        TrackJobBase::size_type const until_turn,
        TrackJobBase::size_type const* SIXTRL_RESTRICT particle_blk_idx_begin,
        TrackJobBase::size_type const  particle_blk_idx_length,
        TrackJobBase::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_index_offset,
        TrackJobBase::size_type* SIXTRL_RESTRICT ptr_beam_monitor_index_offset,
        TrackJobBase::particle_index_t* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        return this->doInitBuffersBaseImpl(
            particles_buffer, belements_buffer,
            output_buffer, num_elem_by_elem_turns, until_turn,
            particle_blk_idx_begin, particle_blk_idx_length,
            ptr_elem_by_elem_index_offset, ptr_beam_monitor_index_offset,
            ptr_min_turn_id );
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

    bool TrackJobBase::doInitBuffersBaseImpl(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT belements_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobBase::size_type const num_elem_by_elem_turns,
        TrackJobBase::size_type const until_turn,
        TrackJobBase::size_type const* SIXTRL_RESTRICT particle_blk_idx_begin,
        TrackJobBase::size_type const  particle_blk_idx_length,
        TrackJobBase::size_type* SIXTRL_RESTRICT ptr_elem_by_elem_index_offset,
        TrackJobBase::size_type* SIXTRL_RESTRICT ptr_beam_monitor_index_offset,
        TrackJobBase::particle_index_t* SIXTRL_RESTRICT ptr_min_turn_id )
    {
        using index_t = TrackJobBase::particle_index_t;
        using size_t  = TrackJobBase::size_type;

        size_t elem_by_elem_index_offset = size_t{ 0 };
        size_t beam_monitor_index_offset = size_t{ 0 };
        index_t min_turn_id = index_t{ -1 };

        bool success = true;

        success &= ( particle_blk_idx_begin  != nullptr );
        success &= ( particle_blk_idx_length == size_t{ 1 } );

        ::NS(Particles)* particles = ::NS(Particles_buffer_get_particles)(
            particles_buffer, particle_blk_idx_begin[ 0 ] );

        success &= ( particles != nullptr );

        success &= ( 0 == NS(OutputBuffer_prepare)( belements_buffer,
            output_buffer, particles, num_elem_by_elem_turns,
            &elem_by_elem_index_offset, &beam_monitor_index_offset,
            &min_turn_id ) );

        success &= ( 0 == NS(BeamMonitor_assign_output_buffer_from_offset)(
            belements_buffer, output_buffer, min_turn_id,
            beam_monitor_index_offset ) );

        if( success )
        {
            if(  ptr_elem_by_elem_index_offset != nullptr )
            {
                *ptr_elem_by_elem_index_offset  = elem_by_elem_index_offset;
            }

            if(  ptr_beam_monitor_index_offset != nullptr )
            {
                *ptr_beam_monitor_index_offset  = beam_monitor_index_offset;
            }

            if(  ptr_min_turn_id != nullptr )
            {
                *ptr_min_turn_id  = min_turn_id;
            }

            this->doSetPtrToParticlesBuffer( particles_buffer );
            this->doSetPtrToBeamElementsBuffer( belements_buffer );
            this->doSetPtrToOutputBuffer( output_buffer );
        }

        return success;
    }
}

#else /* defined( __cplusplus ) */

#endif /* defined( __cplusplus ) */

/* end: sixtracklib/common/internal/track_job_base.cpp */
