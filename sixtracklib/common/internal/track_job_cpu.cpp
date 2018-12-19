#include "sixtracklib/common/track_job_cpu.h"

#include <algorithm>
#include <utility>

#if defined( __cplusplus )
    #include "sixtracklib/common/buffer.hpp"
#endif /* defined( __cplusplus ) */

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/internal/track_job_base.h"
#include "sixtracklib/common/output/output_buffer.h"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/common/track.h"


namespace SIXTRL_CXX_NAMESPACE
{
    TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::size_type const until_turn,
        TrackJobCpu::size_type const num_elem_by_elem_turns  ) :
        TrackJobBase(),
        m_elem_by_elem_index_offset( TrackJobCpu::size_type{ 0 } ),
        m_beam_monitor_index_offset( TrackJobCpu::size_type{ 0 } ),
        m_particle_block_idx( TrackJobCpu::size_type{ 0 } ),
        m_owns_output_buffer( false )

    {
        using size_t  = TrackJobCpu::size_type;
        using index_t = NS(particle_index_t);

        TrackJobCpu::c_buffer_t* output_buffer =
            ::NS(Buffer_new)( size_t{ 0 } );

        index_t min_turn_id = index_t{ -1 };

        ::NS(Particles)* particles = ::NS(Particles_buffer_get_particles)(
            particles_buffer, this->m_particle_block_idx );

        SIXTRL_ASSERT( particles != nullptr );

        int ret = NS(OutputBuffer_prepare)( beam_elements_buffer,
            output_buffer, particles, num_elem_by_elem_turns,
            &this->m_elem_by_elem_index_offset,
            &this->m_beam_monitor_index_offset,
            &min_turn_id );

        SIXTRL_ASSERT( ret == 0 );

        ret = NS(BeamMonitor_assign_output_buffer_from_offset)(
            beam_elements_buffer, output_buffer, min_turn_id,
            this->m_beam_monitor_index_offset );

        SIXTRL_ASSERT( ret == 0 );

        if( num_elem_by_elem_turns > ( size_t{ 0 } ) )
        {
            ::NS(Particles)* elem_by_elem_particles =
                NS(Particles_buffer_get_particles)(
                    output_buffer, this->m_elem_by_elem_index_offset );

            SIXTRL_ASSERT( elem_by_elem_particles != nullptr );

            ret = NS(Track_all_particles_element_by_element_until_turn)(
                particles, beam_elements_buffer,
                min_turn_id + num_elem_by_elem_turns, elem_by_elem_particles );

            SIXTRL_ASSERT( ret == 0 );
        }

        if( until_turn > size_t{ 0 } )
        {
            ret = NS(Track_all_particles_until_turn)(
                particles, beam_elements_buffer, until_turn );

            SIXTRL_ASSERT( ret == 0 );
        }

        this->doSetPtrToParticlesBuffer( particles_buffer );
        this->doSetPtrToBeamElementsBuffer( beam_elements_buffer );
        this->doSetPtrToOutputBuffer( output_buffer );

        this->m_owns_output_buffer = true;

        ( void )ret;
    }

    TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobCpu::size_type const until_turn,
        TrackJobCpu::size_type const num_elem_by_elem_turns  ) :
        TrackJobBase(),
        m_elem_by_elem_index_offset( TrackJobCpu::size_type{ 0 } ),
        m_beam_monitor_index_offset( TrackJobCpu::size_type{ 0 } ),
        m_particle_block_idx( TrackJobCpu::size_type{ 0 } ),
        m_owns_output_buffer( false )

    {
        using size_t  = TrackJobCpu::size_type;
        using index_t = NS(particle_index_t);

        index_t min_turn_id = index_t{ -1 };

        ::NS(Particles)* particles = ::NS(Particles_buffer_get_particles)(
            particles_buffer, this->m_particle_block_idx );

        SIXTRL_ASSERT( particles != nullptr );

        int ret = NS(OutputBuffer_prepare)( beam_elements_buffer,
            output_buffer, particles, num_elem_by_elem_turns,
            &this->m_elem_by_elem_index_offset,
            &this->m_beam_monitor_index_offset,
            &min_turn_id );

        SIXTRL_ASSERT( ret == 0 );

        ret = NS(BeamMonitor_assign_output_buffer_from_offset)(
            beam_elements_buffer, output_buffer, min_turn_id,
            this->m_beam_monitor_index_offset );

        SIXTRL_ASSERT( ret == 0 );

        if( num_elem_by_elem_turns > ( size_t{ 0 } ) )
        {
            ::NS(Particles)* elem_by_elem_particles =
                NS(Particles_buffer_get_particles)(
                    output_buffer, this->m_elem_by_elem_index_offset );

            SIXTRL_ASSERT( elem_by_elem_particles != nullptr );

            ret = NS(Track_all_particles_element_by_element_until_turn)(
                particles, beam_elements_buffer,
                min_turn_id + num_elem_by_elem_turns, elem_by_elem_particles );

            SIXTRL_ASSERT( ret == 0 );
        }

        if( until_turn > size_t{ 0 } )
        {
            ret = NS(Track_all_particles_until_turn)(
                particles, beam_elements_buffer, until_turn );

            SIXTRL_ASSERT( ret == 0 );
        }

        this->doSetPtrToParticlesBuffer( particles_buffer );
        this->doSetPtrToBeamElementsBuffer( beam_elements_buffer );
        this->doSetPtrToOutputBuffer( output_buffer );

        ( void )ret;
    }

    TrackJobCpu::~TrackJobCpu() SIXTRL_NOEXCEPT
    {
        if( this->m_owns_output_buffer)
        {
           ::NS(Buffer_delete)( this->doGetPtrOutputBuffer() );
           this->doSetPtrToOutputBuffer( nullptr );
        }
    }

    TrackJobCpu::c_buffer_t*
    TrackJobCpu::track( TrackJobCpu::size_type const until_turn )
    {
        SIXTRL_ASSERT( this->doGetPtrParticlesBuffer()    != nullptr );
        SIXTRL_ASSERT( this->doGetPtrBeamElementsBuffer() != nullptr );
        SIXTRL_ASSERT( this->doGetPtrOutputBuffer()       != nullptr );

        ::NS(Particles)* particles = ::NS(Particles_buffer_get_particles)(
            this->doGetPtrParticlesBuffer(), this->m_particle_block_idx );

        SIXTRL_ASSERT( particles != nullptr );

        int ret = NS(Track_all_particles_until_turn)(
            particles, this->doGetPtrBeamElementsBuffer(), until_turn );

        ( void )ret;

        return this->doGetPtrOutputBuffer();
    }

    void TrackJobCpu::collect()
    {
        return;
    }

}


SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn )
{
    using track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCpu;

    track_job_t* track_job = new track_job_t(
        particles_buffer, beam_elements_buffer,
        num_elem_by_elem_turns, until_turn );

    return track_job;
}


SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new_using_output_buffer)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(buffer_size_t) const until_turn )
{
    using track_job_t = SIXTRL_CXX_NAMESPACE::TrackJobCpu;

    track_job_t* track_job = new track_job_t(
        particles_buffer, beam_elements_buffer, output_buffer,
        num_elem_by_elem_turns, until_turn );

    return track_job;
}

SIXTRL_HOST_FN void NS(TrackJobCpu_delete)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job )
{
    delete track_job;
    return;
}

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCpu_track)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job,
    NS(buffer_size_t) const until_turn )
{
    return ( track_job != nullptr ) ? track_job->track( until_turn ) : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN void NS(TrackJobCpu_collect)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job )
{
    if( track_job != nullptr )
    {
        track_job->collect();
    }

    return;
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCpu_get_particle_buffer)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrParticlesBuffer() : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)* NS(TrackJobCpu_get_output_buffer)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr ) ? track_job->ptrOutputBuffer() : nullptr;
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJobCpu_get_beam_elements_buffer)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT track_job )
{
    return ( track_job != nullptr )
        ? track_job->ptrBeamElementsBuffer() : nullptr;
}

/* end: sixtracklib/common/internal/track_job_cpu.cpp */
