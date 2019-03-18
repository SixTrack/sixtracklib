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
    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu( std::string const& config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        if( !config_str.empty() )
        {
            TrackJobBase::doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( config_str.c_str() );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu(
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > TrackJobCpu::size_type{ 0 } ) )
        {
            TrackJobBase::doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr(  config_str );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::size_type const target_num_output_turns,
        TrackJobCpu::size_type const num_elem_by_elem_turns,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        if( TrackJobBase::doReset(
            particles_buffer, beam_elements_buffer, ptr_output_buffer,
            target_num_output_turns, num_elem_by_elem_turns ) )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elements_buffer );
        }


        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > TrackJobCpu::size_type{ 0 } ) )
        {
            TrackJobBase::doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr( config_str );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCpu::size_type const num_particle_sets,
        TrackJobCpu::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::size_type const target_num_output_turns,
        TrackJobCpu::size_type const num_elem_by_elem_turns,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        if( ( num_particle_sets > TrackJobCpu::size_type{ 0 } ) &&
            ( pset_indices_begin != nullptr ) )
        {
            TrackJobCpu::size_type const*
                pset_indices_end = pset_indices_begin;

            std::advance( pset_indices_end, num_particle_sets );

            this->doSetParticleSetIndices(
                pset_indices_begin, pset_indices_end );
        }

        if( TrackJobBase::doReset(
                particles_buffer, beam_elements_buffer, ptr_output_buffer,
                target_num_output_turns, num_elem_by_elem_turns ) )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elements_buffer );
        }

        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > TrackJobCpu::size_type{ 0 } ) )
        {
            TrackJobBase::doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr( config_str );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu(
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        TrackJobCpu::size_type const target_num_output_turns,
        TrackJobCpu::size_type const num_elem_by_elem_turns,
        TrackJobCpu::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        std::string const& config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        if( TrackJobBase::doReset( particles_buffer.getCApiPtr(),
            beam_elements_buffer.getCApiPtr(), ( ptr_output_buffer != nullptr )
                ? ptr_output_buffer->getCApiPtr() : nullptr,
            target_num_output_turns, num_elem_by_elem_turns ) )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( ptr_output_buffer != nullptr ) &&
                ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }

        if( !config_str.empty() )
        {
            TrackJobBase::doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( config_str.c_str() );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::~TrackJobCpu() SIXTRL_NOEXCEPT {}

    SIXTRL_HOST_FN TrackJobCpu::track_status_t TrackJobCpu::doTrackUntilTurn(
        TrackJobCpu::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::track( *this, until_turn );
    }

    SIXTRL_HOST_FN TrackJobCpu::track_status_t TrackJobCpu::doTrackElemByElem(
        TrackJobCpu::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackElemByElem( *this, until_turn );
    }

    SIXTRL_HOST_FN void TrackJobCpu::doCollect()
    {
        SIXTRL_CXX_NAMESPACE::collect( *this );
        return;
    }

    SIXTRL_HOST_FN TrackJobCpu::track_status_t track(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        TrackJobCpu::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using size_t             = TrackJobCpu::size_type;
        using ptr_particles_t    = SIXTRL_PARTICLE_ARGPTR_DEC ::NS(Particles)*;
        using ptr_part_buffer_t  = SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)*;
        using ptr_belem_buffer_t = SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)*;
        using track_status_t     = TrackJobCpu::track_status_t;
        using particle_index_t   = TrackJobCpu::particle_index_t;

        track_status_t     status       = track_status_t{ -1 };
        ptr_part_buffer_t  pbuffer      = job.ptrCParticlesBuffer();
        ptr_belem_buffer_t belem_buffer = job.ptrCBeamElementsBuffer();

        particle_index_t const _until_turn_num =
            static_cast< particle_index_t >( until_turn );

        SIXTRL_ASSERT( pbuffer      != nullptr );
        SIXTRL_ASSERT( belem_buffer != nullptr );

        if( job.numParticleSets() == size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );

            ptr_particles_t particles = ::NS(Particles_buffer_get_particles)(
                pbuffer, *job.particleSetIndicesBegin() );

            status = ::NS(Track_all_particles_until_turn)(
                particles, belem_buffer, _until_turn_num );
        }
        else if( job.numParticleSets() > size_t{ 1 } )
        {
            size_t const* pset_it    = job.particleSetIndicesBegin();
            size_t const* pset_end   = job.particleSetIndicesEnd();

            SIXTRL_ASSERT( pset_it  != nullptr );
            SIXTRL_ASSERT( pset_end != nullptr );

            status = track_status_t{ 0 };

            for(  ; pset_it != pset_end ; ++pset_it )
            {
                status |= NS(Track_all_particles_until_turn)(
                    ::NS(Particles_buffer_get_particles)( pbuffer, *pset_it ),
                    belem_buffer, _until_turn_num );
            }
        }

        return status;
    }

    SIXTRL_HOST_FN TrackJobCpu::track_status_t trackElemByElem(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        TrackJobCpu::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using size_t                = TrackJobCpu::size_type;
        using ptr_particles_t       = SIXTRL_PARTICLE_ARGPTR_DEC ::NS(Particles)*;
        using ptr_part_buffer_t     = SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)*;
        using ptr_belem_buffer_t    = SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)*;
        using track_status_t        = TrackJobCpu::track_status_t;
        using elem_by_elem_config_t = TrackJobBase::elem_by_elem_config_t;

        track_status_t     status       = track_status_t{ -1 };
        ptr_part_buffer_t  pbuffer      = job.ptrCParticlesBuffer();
        ptr_belem_buffer_t belem_buffer = job.ptrCBeamElementsBuffer();

        SIXTRL_ASSERT( pbuffer      != nullptr );
        SIXTRL_ASSERT( belem_buffer != nullptr );

        if( ( job.hasOutputBuffer() ) && ( job.hasElemByElemOutput() ) &&
            ( job.hasElemByElemConfig() ) )
        {
            TrackJobCpu const& cjob      = job;
            ptr_part_buffer_t out_buffer = job.ptrCOutputBuffer();

            size_t const* pset_it    = cjob.particleSetIndicesBegin();
            size_t const* pset_end   = cjob.particleSetIndicesEnd();

            SIXTRL_ASSERT( out_buffer != nullptr );
            SIXTRL_ASSERT( pset_it    != nullptr );
            SIXTRL_ASSERT( pset_end   != nullptr );

            elem_by_elem_config_t const* elem_by_elem_config =
                cjob.ptrElemByElemConfig();

            auto be_begin = ::NS(Buffer_get_const_objects_begin)(belem_buffer);
            auto be_end = ::NS(Buffer_get_const_objects_end)( belem_buffer );

            status = track_status_t{ 0 };

            for( ; pset_it != pset_end ; ++pset_it )
            {
                ptr_particles_t particles =
                    ::NS(Particles_buffer_get_particles)( pbuffer, *pset_it );

                status |= ::NS(Track_all_particles_element_by_element_until_turn_objs)(
                    particles, elem_by_elem_config, be_begin, be_end,
                        until_turn );
            }
        }

        return status;
    }
}

SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_create)( void )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( nullptr );
}

SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_create_from_config_str)(
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( config_str );
}

SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const max_output_turns,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    NS(Buffer)* ptr_output_buffer = nullptr;
    char const* config_str = nullptr;

    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu(
        particles_buffer, beam_elements_buffer, max_output_turns,
        num_elem_by_elem_turns, ptr_output_buffer, config_str );
}

SIXTRL_EXTERN SIXTRL_HOST_FN NS(TrackJobCpu)*
NS(TrackJobCpu_new_for_particle_sets)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const max_output_turns,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    NS(Buffer)* ptr_output_buffer = nullptr;
    char const* config_str = nullptr;

    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu(
        particles_buffer, num_particle_sets, particle_set_indices_begin,
        beam_elements_buffer, max_output_turns,
        num_elem_by_elem_turns, ptr_output_buffer, config_str );
}

SIXTRL_HOST_FN NS(TrackJobCpu)* NS(TrackJobCpu_new_detailed)(
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const max_output_turns,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu(
        particles_buffer, num_particle_sets, particle_set_indices_begin,
        beam_elements_buffer, max_output_turns, num_elem_by_elem_turns,
        output_buffer, config_str );
}

SIXTRL_HOST_FN void NS(TrackJobCpu_delete)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    delete job;
    return;
}

SIXTRL_HOST_FN void NS(TrackJobCpu_clear)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->clear();
}

SIXTRL_HOST_FN bool NS(TrackJobCpu_reset)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const max_output_turns,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elements_buffer,
                      max_output_turns, num_elem_by_elem_turns, nullptr )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJobCpu_reset_detailed)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    NS(buffer_size_t) const max_output_turns,
    NS(buffer_size_t) const num_elem_by_elem_turns,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer )
{
    bool success = false;

    if( job != nullptr )
    {
        ::NS(buffer_size_t) const* particle_set_indices_end =
            particle_set_indices_begin;

        if( ( num_particle_sets > ::NS(buffer_size_t){ 0 } ) &&
            ( particle_set_indices_begin != nullptr ) )
        {
            std::advance( particle_set_indices_end, num_particle_sets );
        }

        success = job->reset( particles_buffer,
            particle_set_indices_begin, particle_set_indices_end,
            beam_elements_buffer, max_output_turns, num_elem_by_elem_turns,
            output_buffer );
    }

    return success;
}

SIXTRL_HOST_FN bool NS(TrackJobCpu_assign_output_buffer)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const output_buffer_offset )
{
    return ( job != nullptr )
        ? job->assignOutputBuffer( output_buffer ) : false;
}

SIXTRL_HOST_FN void NS(TrackJobCpu_collect)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    SIXTRL_CXX_NAMESPACE::collect( *job );

    return;
}

SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCpu_track_until_turn)( NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const until_turn )
{
    SIXTRL_ASSERT( job != nullptr );
    return SIXTRL_CXX_NAMESPACE::track( *job, until_turn );
}

SIXTRL_HOST_FN NS(track_status_t)
NS(TrackJobCpu_track_elem_by_elem)( NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const num_elem_by_elem_turns )
{
    SIXTRL_ASSERT( job != nullptr );
    return SIXTRL_CXX_NAMESPACE::trackElemByElem(
        *job, num_elem_by_elem_turns );
}

/* end: sixtracklib/common/internal/track_job_cpu.cpp */
