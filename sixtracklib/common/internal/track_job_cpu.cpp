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
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        if( TrackJobBase::doReset( particles_buffer, beam_elements_buffer,
                ptr_output_buffer, until_turn_elem_by_elem ) )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elements_buffer );
        }

        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > TrackJobCpu::size_type{ 0 } ) )
        {
            TrackJobBase::doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCpu::size_type const num_particle_sets,
        TrackJobCpu::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        using size_t = TrackJobCpu::size_type;
        size_t const* pset_indices_end = pset_indices_begin;

        if( ( num_particle_sets > size_t{ 0 } ) &&
            ( pset_indices_end != nullptr ) )
        {
            std::advance( pset_indices_end, num_particle_sets );
            this->doSetParticleSetIndices( pset_indices_begin,
                pset_indices_end, particles_buffer );
        }

        if( TrackJobBase::doReset( particles_buffer, beam_elements_buffer,
                ptr_output_buffer, until_turn_elem_by_elem ) )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elements_buffer );
        }

        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > size_t{ 0 } ) )
        {
            TrackJobBase::doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu(
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        TrackJobCpu::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const  until_turn_elem_by_elem,
        std::string const& config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        using c_buffer_t = TrackJobCpu::c_buffer_t;

        c_buffer_t* ptr_part_buffer  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_belem_buffer = beam_elements_buffer.getCApiPtr();
        c_buffer_t* ptr_out_buffer   = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        if( TrackJobBase::doReset( ptr_part_buffer, ptr_belem_buffer,
                ptr_out_buffer, until_turn_elem_by_elem ) )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( ptr_out_buffer != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }

        if( !config_str.empty() )
        {
            TrackJobBase::doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
        }
    }

    SIXTRL_HOST_FN TrackJobCpu::TrackJobCpu(
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobCpu::size_type const num_particle_sets,
        TrackJobCpu::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        TrackJobCpu::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        TrackJobCpu::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const until_turn_elem_by_elem,
        std::string const& config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        using size_t = TrackJobCpu::size_type;
        using c_buffer_t = TrackJobCpu::c_buffer_t;

        c_buffer_t* ptr_part_buffer  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_belem_buffer = beam_elements_buffer.getCApiPtr();
        c_buffer_t* ptr_out_buffer   = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        size_t const* pset_indices_end = pset_indices_begin;

        if( ( pset_indices_end != nullptr ) &&
            ( num_particle_sets > size_t{ 0 } ) )
        {
            std::advance( pset_indices_end, num_particle_sets );

            this->doSetParticleSetIndices(
                pset_indices_begin, pset_indices_end, ptr_part_buffer );
        }

        if( TrackJobBase::doReset( ptr_part_buffer, ptr_belem_buffer,
                ptr_out_buffer, until_turn_elem_by_elem ) )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( this->hasOutputBuffer() ) && ( ptr_out_buffer != nullptr ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }

        if( !config_str.empty() )
        {
            TrackJobBase::doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( this->ptrConfigStr() );
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

    SIXTRL_HOST_FN TrackJobCpu::track_status_t TrackJobCpu::doTrackLine(
        TrackJobCpu::size_type const beam_elements_begin_index,
        TrackJobCpu::size_type const beam_elements_end_index,
        bool const finish_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackLine( *this,
            beam_elements_begin_index, beam_elements_end_index, finish_turn );
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
            TrackJobCpu const& cjob  = job;
            size_t const* pset_it    = cjob.particleSetIndicesBegin();
            size_t const* pset_end   = cjob.particleSetIndicesEnd();

            SIXTRL_ASSERT( job.ptrCOutputBuffer() != nullptr );
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

    SIXTRL_HOST_FN TrackJobCpu::track_status_t trackLine(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        TrackJobCpu::size_type const belem_begin_index,
        TrackJobCpu::size_type const belem_end_index,
        bool const finish_turn ) SIXTRL_NOEXCEPT
    {
        using size_t                = TrackJobCpu::size_type;
        using ptr_particles_t       = SIXTRL_PARTICLE_ARGPTR_DEC ::NS(Particles)*;
        using ptr_part_buffer_t     = SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)*;
        using ptr_belem_buffer_t    = SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)*;
        using track_status_t        = TrackJobCpu::track_status_t;

        track_status_t ret = track_status_t{ -1 };

        ptr_part_buffer_t  pbuffer      = job.ptrCParticlesBuffer();
        ptr_belem_buffer_t belem_buffer = job.ptrCBeamElementsBuffer();

        size_t const num_beam_elements =
            ::NS(Buffer_get_num_of_objects)( belem_buffer );

        if( ( belem_begin_index < belem_end_index ) &&
            ( belem_end_index <= num_beam_elements ) )
        {
            if( job.numParticleSets() == size_t{ 1 } )
            {
                SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );

                ptr_particles_t p = ::NS(Particles_buffer_get_particles)(
                    pbuffer, *job.particleSetIndicesBegin() );

                ret = ::NS(Track_all_particles_subset_of_beam_elements)( p,
                    belem_buffer, belem_begin_index, belem_end_index );

                if( ( finish_turn ) && ( ret == track_status_t{ 0 } ) )
                {
                    ::NS(Particles_increment_all_at_turn_values)( p );
                    ::NS(Particles_set_all_at_element_id_value)( p,
                        job.minElementId() );
                }
            }
            else if( job.numParticleSets() > size_t{ 1 }  )
            {
                size_t const* it    = job.particleSetIndicesBegin();
                size_t const* end   = job.particleSetIndicesEnd();

                SIXTRL_ASSERT( it  != nullptr );
                SIXTRL_ASSERT( end != nullptr );

                ret = track_status_t{ 0 };

                for( ; it != end ; ++it )
                {
                    ptr_particles_t p =
                        ::NS(Particles_buffer_get_particles)( pbuffer, *it );

                    ret |= ::NS(Track_all_particles_subset_of_beam_elements)(
                        p, belem_buffer, belem_begin_index, belem_end_index );

                    if( ( finish_turn ) && ( ret == track_status_t{ 0 } ) )
                    {
                        ::NS(Particles_increment_all_at_turn_values)( p );
                        ::NS(Particles_set_all_at_element_id_value)( p,
                            job.minElementId() );
                    }
                }
            }
        }

        return ret;
    }
}

SIXTRL_HOST_FN ::NS(TrackJobCpu)* NS(TrackJobCpu_create)( void )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( nullptr );
}

SIXTRL_HOST_FN ::NS(TrackJobCpu)* NS(TrackJobCpu_create_from_config_str)(
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( config_str );
}

SIXTRL_HOST_FN ::NS(TrackJobCpu)* NS(TrackJobCpu_new)(
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer )
{
    using _this_t    = SIXTRL_CXX_NAMESPACE::TrackJobCpu;
    using c_buffer_t = _this_t::c_buffer_t;
    using size_t     = _this_t::size_type;

    c_buffer_t* ptr_output_buffer = nullptr;
    char const* config_str = nullptr;

    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( particles_buffer,
        beam_elements_buffer, ptr_output_buffer, size_t{ 0 }, config_str );
}

SIXTRL_HOST_FN ::NS(TrackJobCpu)* NS(TrackJobCpu_new_with_output)(
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( particles_buffer,
        beam_elements_buffer, output_buffer, until_turn_elem_by_elem );
}

SIXTRL_HOST_FN ::NS(TrackJobCpu)* NS(TrackJobCpu_new_detailed)(
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const num_particle_sets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem,
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( particles_buffer,
        num_particle_sets, particle_set_indices_begin, beam_elements_buffer,
        output_buffer, until_turn_elem_by_elem, config_str );
}

SIXTRL_HOST_FN void NS(TrackJobCpu_delete)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    delete job;
    return;
}

SIXTRL_HOST_FN void NS(TrackJobCpu_clear)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->clear();
}

SIXTRL_HOST_FN bool NS(TrackJobCpu_reset)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elements_buffer, output_buffer )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJobCpu_reset_with_output)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elements_buffer,
                      output_buffer, until_turn_elem_by_elem )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJobCpu_reset_detailed)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(buffer_size_t) const num_particle_sets,
    ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, num_particle_sets, pset_indices_begin,
            beam_elements_buffer, output_buffer, until_turn_elem_by_elem )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJobCpu_assign_output_buffer)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT out_buffer )
{
    return ( job != nullptr ) ? job->assignOutputBuffer( out_buffer ) : false;
}

SIXTRL_HOST_FN void NS(TrackJobCpu_collect)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    SIXTRL_CXX_NAMESPACE::collect( *job );

    return;
}

SIXTRL_HOST_FN ::NS(track_status_t) NS(TrackJobCpu_track_until_turn)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job, ::NS(buffer_size_t) const turn )
{
    SIXTRL_ASSERT( job != nullptr );
    return SIXTRL_CXX_NAMESPACE::track( *job, turn );
}

SIXTRL_HOST_FN ::NS(track_status_t) NS(TrackJobCpu_track_elem_by_elem)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job, ::NS(buffer_size_t) const turn )
{
    SIXTRL_ASSERT( job != nullptr );
    return SIXTRL_CXX_NAMESPACE::trackElemByElem( *job, turn );
}

/* end: sixtracklib/common/internal/track_job_cpu.cpp */
