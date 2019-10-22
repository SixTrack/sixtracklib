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
#include "sixtracklib/common/track/track.h"
#include "sixtracklib/common/track/track_kernel_impl.h"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        using _this_t = st::TrackJobCpu;
    }

    TrackJobCpu::TrackJobCpu( std::string const& config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        this->doSetRequiresCollectFlag( false );

        if( !config_str.empty() )
        {
            TrackJobBase::doSetConfigStr( config_str.c_str() );
            TrackJobBase::doParseConfigStr( config_str.c_str() );
        }
    }

    TrackJobCpu::TrackJobCpu(
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        this->doSetRequiresCollectFlag( false );

        if( ( config_str != nullptr ) &&
            ( std::strlen( config_str ) > TrackJobCpu::size_type{ 0 } ) )
        {
            TrackJobBase::doSetConfigStr( config_str );
            TrackJobBase::doParseConfigStr(  config_str );
        }
    }

    TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        this->doSetRequiresCollectFlag( false );

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

    TrackJobCpu::TrackJobCpu(
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobCpu::size_type const particle_set_index,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobCpu::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobCpu::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
        TrackJobBase( SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_STR,
                      SIXTRL_CXX_NAMESPACE::TRACK_JOB_CPU_ID )
    {
        using size_t = TrackJobCpu::size_type;
        this->doSetRequiresCollectFlag( false );

        size_t particle_set_indices[ 2 ] = { size_t{ 0 }, size_t{ 0 } };

        particle_set_indices[ 0 ] =
        particle_set_indices[ 1 ] = particle_set_index;
        this->doSetParticleSetIndices(
            &particle_set_indices[ 0 ], &particle_set_indices[ 1 ] );

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

    TrackJobCpu::TrackJobCpu(
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

        this->doSetRequiresCollectFlag( false );

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

    TrackJobCpu::TrackJobCpu(
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

        this->doSetRequiresCollectFlag( false );

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

    TrackJobCpu::TrackJobCpu(
        buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        size_type const particle_set_index,
        buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem,
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

        this->doSetRequiresCollectFlag( false );
        size_t particle_set_indices[ 2 ] = { size_t{ 0 }, size_t{ 0 } };
        particle_set_indices[ 0 ] = particle_set_indices[ 1 ] =
            particle_set_index;

        this->doSetParticleSetIndices(
            &particle_set_indices[ 0 ], &particle_set_indices[ 1 ] );

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


    TrackJobCpu::TrackJobCpu(
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

        this->doSetRequiresCollectFlag( false );
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

    TrackJobCpu::~TrackJobCpu() SIXTRL_NOEXCEPT {}

    TrackJobCpu::track_status_t TrackJobCpu::doTrackUntilTurn(
        TrackJobCpu::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackUntil( *this, until_turn );
    }

    TrackJobCpu::track_status_t TrackJobCpu::doTrackElemByElem(
        TrackJobCpu::size_type const until_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackElemByElem( *this, until_turn );
    }

    TrackJobCpu::track_status_t TrackJobCpu::doTrackLine(
        TrackJobCpu::size_type const beam_elements_begin_index,
        TrackJobCpu::size_type const beam_elements_end_index,
        bool const finish_turn )
    {
        return SIXTRL_CXX_NAMESPACE::trackLine( *this,
            beam_elements_begin_index, beam_elements_end_index, finish_turn );
    }

    void TrackJobCpu::doCollect(
        TrackJobCpu::collect_flag_t const flags )
    {
        SIXTRL_CXX_NAMESPACE::collect( *this, flags );
        return;
    }

    TrackJobCpu::track_status_t trackUntil(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        TrackJobCpu::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using size_t   = _this_t::size_type;
        using pindex_t = _this_t::particle_index_t;

        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        pindex_t const _until_turn_num = static_cast< pindex_t >( until_turn );

        SIXTRL_ASSERT( job.ptrCParticlesBuffer() != nullptr );
        SIXTRL_ASSERT( job.ptrCBeamElementsBuffer() != nullptr );

        if( job.numParticleSets() == size_t{ 1 } )
        {
            status = ::NS(Track_particles_until_turn_kernel_impl)(
                ::NS(Buffer_get_data_begin)( job.ptrCParticlesBuffer() ),
                *job.particleSetIndicesBegin(), pindex_t{ 0 }, pindex_t{ 1 },
                ::NS(Buffer_get_data_begin)( job.ptrCBeamElementsBuffer() ),
                _until_turn_num, ::NS(Buffer_get_slot_size)(
                    job.ptrCParticlesBuffer() ) );
        }
        else if( job.numParticleSets() > size_t{ 1 } )
        {
            size_t const* pset_it  = job.particleSetIndicesBegin();
            size_t const* pset_end = job.particleSetIndicesEnd();

            SIXTRL_ASSERT( pset_it  != nullptr );
            SIXTRL_ASSERT( pset_end != nullptr );

            status = st::TRACK_SUCCESS;

            for(  ; pset_it != pset_end ; ++pset_it )
            {
                status |= ::NS(Track_particles_until_turn_kernel_impl)(
                    ::NS(Buffer_get_data_begin)( job.ptrCParticlesBuffer() ),
                    *pset_it, pindex_t{ 0 }, pindex_t{ 1 },
                    ::NS(Buffer_get_data_begin)( job.ptrCBeamElementsBuffer() ),
                    _until_turn_num,
                    ::NS(Buffer_get_slot_size)( job.ptrCParticlesBuffer() ) );
            }
        }

        return status;
    }

    _this_t::track_status_t trackElemByElem(
        TrackJobCpu& SIXTRL_RESTRICT_REF job,
        _this_t::size_type const until_turn ) SIXTRL_NOEXCEPT
    {
        using size_t   = _this_t::size_type;
        using pindex_t = _this_t::particle_index_t;

        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;
        pindex_t const _until_turn_num = static_cast< pindex_t >( until_turn );

        if( ( job.hasOutputBuffer() ) && ( job.hasElemByElemOutput() ) &&
            ( job.hasElemByElemConfig() ) )
        {
            SIXTRL_ASSERT( job.ptrCParticlesBuffer()    != nullptr );
            SIXTRL_ASSERT( job.ptrCBeamElementsBuffer() != nullptr );
            SIXTRL_ASSERT( job.ptrElemByElemConfig()    != nullptr );

            size_t const slot_size  = ::NS(Buffer_get_slot_size)(
                    job.ptrCParticlesBuffer() );

            if( job.numParticleSets() == size_t{ 1 } )
            {
                SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );

                status = NS(Track_particles_elem_by_elem_until_turn_kernel_impl)(
                    ::NS(Buffer_get_data_begin)( job.ptrCParticlesBuffer() ),
                    *job.particleSetIndicesBegin(), pindex_t{ 0 }, pindex_t{ 1 },
                    ::NS(Buffer_get_data_begin)( job.ptrCBeamElementsBuffer() ),
                    job.ptrElemByElemConfig(), _until_turn_num, slot_size );
            }
            else if( job.numParticleSets() > size_t{ 1 } )
            {
                TrackJobCpu const& cjob = job;
                size_t const* pset_it   = cjob.particleSetIndicesBegin();
                size_t const* pset_end  = cjob.particleSetIndicesEnd();

                SIXTRL_ASSERT( pset_it  != nullptr );
                SIXTRL_ASSERT( pset_end != nullptr );
                SIXTRL_ASSERT( std::distance( pset_it, pset_end ) >= 0 );

                status = st::TRACK_SUCCESS;

                for( ; pset_it != pset_end ; ++pset_it )
                {
                    status |=
                    NS(Track_particles_elem_by_elem_until_turn_kernel_impl)(
                        ::NS(Buffer_get_data_begin)( job.ptrCParticlesBuffer() ),
                        *pset_it, pindex_t{ 0 }, pindex_t{ 1 },
                        ::NS(Buffer_get_data_begin)(
                            job.ptrCBeamElementsBuffer() ),
                        job.ptrElemByElemConfig(), _until_turn_num, slot_size );
                }
            }
        }

        return status;
    }

    _this_t::track_status_t trackLine( TrackJobCpu& SIXTRL_RESTRICT_REF job,
        _this_t::size_type const begin_idx, _this_t::size_type const end_idx,
        bool const finish_turn ) SIXTRL_NOEXCEPT
    {
        using size_t   = _this_t::size_type;
        using pindex_t = _this_t::particle_index_t;

        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        size_t const slot_size = ::NS(Buffer_get_slot_size)(
                job.ptrCParticlesBuffer() );

        if( job.numParticleSets() == size_t{ 1 } )
        {
            SIXTRL_ASSERT( job.particleSetIndicesBegin() != nullptr );

            status = NS(Track_particles_line_kernel_impl)(
                ::NS(Buffer_get_data_begin)( job.ptrCParticlesBuffer() ),
                *job.particleSetIndicesBegin(), pindex_t{ 0 }, pindex_t{ 1 },
                ::NS(Buffer_get_data_begin)( job.ptrCBeamElementsBuffer() ),
                begin_idx, end_idx, finish_turn, slot_size );
        }
        else if( job.numParticleSets() > size_t{ 1 } )
        {
            TrackJobCpu const& cjob = job;
            size_t const* pset_it   = cjob.particleSetIndicesBegin();
            size_t const* pset_end  = cjob.particleSetIndicesEnd();

            SIXTRL_ASSERT( pset_it  != nullptr );
            SIXTRL_ASSERT( pset_end != nullptr );
            SIXTRL_ASSERT( std::distance( pset_it, pset_end ) >= 0 );

            status = st::TRACK_SUCCESS;

            for( ; pset_it != pset_end ; ++pset_it )
            {
                status |= status = NS(Track_particles_line_kernel_impl)(
                ::NS(Buffer_get_data_begin)( job.ptrCParticlesBuffer() ),
                *pset_it, pindex_t{ 0 }, pindex_t{ 1 },
                ::NS(Buffer_get_data_begin)( job.ptrCBeamElementsBuffer() ),
                begin_idx, end_idx, finish_turn, slot_size );
            }
        }

        return status;
    }
}

::NS(TrackJobCpu)* NS(TrackJobCpu_create)( void )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( nullptr );
}

::NS(TrackJobCpu)* NS(TrackJobCpu_create_from_config_str)(
    const char *const SIXTRL_RESTRICT config_str )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( config_str );
}

::NS(TrackJobCpu)* NS(TrackJobCpu_new)(
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

::NS(TrackJobCpu)* NS(TrackJobCpu_new_with_output)(
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    ::NS(buffer_size_t) const until_turn_elem_by_elem )
{
    return new SIXTRL_CXX_NAMESPACE::TrackJobCpu( particles_buffer,
        beam_elements_buffer, output_buffer, until_turn_elem_by_elem );
}

::NS(TrackJobCpu)* NS(TrackJobCpu_new_detailed)(
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

void NS(TrackJobCpu_delete)( ::NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    delete job;
    return;
}

void NS(TrackJobCpu_clear)( ::NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->clear();
}

bool NS(TrackJobCpu_reset)( ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT beam_elements_buffer,
    ::NS(Buffer)* SIXTRL_RESTRICT output_buffer )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elements_buffer, output_buffer )
        : false;
}

bool NS(TrackJobCpu_reset_with_output)( ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
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

bool NS(TrackJobCpu_reset_detailed)( ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
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

bool NS(TrackJobCpu_assign_output_buffer)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    ::NS(Buffer)* SIXTRL_RESTRICT out_buffer )
{
    return ( job != nullptr ) ? job->assignOutputBuffer( out_buffer ) : false;
}

void NS(TrackJobCpu_collect)( ::NS(TrackJobCpu)* SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    SIXTRL_CXX_NAMESPACE::collect( *job );
}

void NS(TrackJobCpu_collect_detailed)( NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(track_job_collect_flag_t) const flags )
{
    SIXTRL_ASSERT( job != nullptr );
    SIXTRL_CXX_NAMESPACE::collect( *job, flags );
}

::NS(track_status_t) NS(TrackJobCpu_track_until_turn)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job, ::NS(buffer_size_t) const turn )
{
    SIXTRL_ASSERT( job != nullptr );
    return SIXTRL_CXX_NAMESPACE::trackUntil( *job, turn );
}

::NS(track_status_t) NS(TrackJobCpu_track_elem_by_elem)(
    ::NS(TrackJobCpu)* SIXTRL_RESTRICT job, ::NS(buffer_size_t) const turn )
{
    SIXTRL_ASSERT( job != nullptr );
    return SIXTRL_CXX_NAMESPACE::trackElemByElem( *job, turn );
}

NS(track_status_t) NS(TrackJobCpu_track_line)(
    NS(TrackJobCpu)* SIXTRL_RESTRICT job,
    NS(buffer_size_t) const begin_index, NS(buffer_size_t) const end_index,
    bool const finish_turn )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->trackLine( begin_index, end_index, finish_turn );
}

/* end: sixtracklib/common/internal/track_job_cpu.cpp */
