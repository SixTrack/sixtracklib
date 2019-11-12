#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include "sixtracklib/common/track/track_job_cpu.hpp"

    #if defined( __cplusplus )
        #include <algorithm>
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
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/debug_register.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/track/track.h"
    #include "sixtracklib/common/track/track_kernel_impl.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/track/track_job_base.hpp"
    #endif /* defined( __cplusplus ) */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        using _this_t = st::CpuTrackJob;
    }

    _this_t::track_status_t trackUntil( CpuTrackJob& SIXTRL_RESTRICT_REF job,
        _this_t::size_type const until_turn )
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

    _this_t::track_status_t trackElemByElem( CpuTrackJob& SIXTRL_RESTRICT_REF job,
        _this_t::size_type const until_turn )
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

            size_t const slot_size = ::NS(Buffer_get_slot_size)(
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
                CpuTrackJob const& cjob = job;
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

    _this_t::track_status_t trackLine( CpuTrackJob& SIXTRL_RESTRICT_REF job,
        _this_t::size_type const begin_idx, _this_t::size_type const end_idx,
        bool const finish_turn )
    {
        using size_t   = _this_t::size_type;
        using pindex_t = _this_t::particle_index_t;

        _this_t::track_status_t status = st::TRACK_STATUS_GENERAL_FAILURE;

        SIXTRL_ASSERT( job.ptrCParticlesBuffer()    != nullptr );
        SIXTRL_ASSERT( job.ptrCBeamElementsBuffer() != nullptr );

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
            CpuTrackJob const& cjob = job;
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

    /* ********************************************************************* */

    CpuTrackJob::CpuTrackJob( const char *const SIXTRL_RESTRICT config_str ) :
        st::TrackJobBaseNew( st::ARCHITECTURE_CPU,
            SIXTRL_ARCHITECTURE_CPU_STR, config_str )
    {
        this->doSetRequiresCollectFlag( false );
    }

    CpuTrackJob::CpuTrackJob(
        std::string const& SIXTRL_RESTRICT_REF config_str ) :
            st::TrackJobBaseNew( st::ARCHITECTURE_CPU,
                SIXTRL_ARCHITECTURE_CPU_STR, config_str.c_str() )
    {
        this->doSetRequiresCollectFlag( false );
    }

    CpuTrackJob::CpuTrackJob(
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CpuTrackJob::size_type const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str ) :
            st::TrackJobBaseNew( st::ARCHITECTURE_CPU,
                SIXTRL_ARCHITECTURE_CPU_STR, config_str )
    {
        this->doSetRequiresCollectFlag( false );

        CpuTrackJob::status_t const status = st::TrackJobBaseNew::doReset(
            particles_buffer, beam_elements_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    CpuTrackJob::CpuTrackJob(
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        CpuTrackJob::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CpuTrackJob::size_type const until_turn_elem_by_elem,
        std::string const& config_str ) : st::TrackJobBaseNew(
            st::ARCHITECTURE_CPU, SIXTRL_ARCHITECTURE_CPU_STR,
                config_str.c_str() )
    {
        this->doSetRequiresCollectFlag( false );

        CpuTrackJob::status_t const status = st::TrackJobBaseNew::doReset(
            particles_buffer.getCApiPtr(), beam_elements_buffer.getCApiPtr(),
            ( ptr_output_buffer != nullptr )
                ? ptr_output_buffer->getCApiPtr() : nullptr,
            until_turn_elem_by_elem );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetCxxBufferPointers(
                particles_buffer, beam_elements_buffer, ptr_output_buffer );
        }
    }

    CpuTrackJob::CpuTrackJob(
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        CpuTrackJob::size_type const particle_set_index,
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        CpuTrackJob::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CpuTrackJob::size_type const until_turn_elem_by_elem,
        std::string const& config_str ) :
            st::TrackJobBaseNew( st::ARCHITECTURE_CPU,
                SIXTRL_ARCHITECTURE_CPU_STR, config_str.c_str() )
    {
        using size_t = CpuTrackJob::size_type;

        size_t const* pset_begin = &particle_set_index;
        size_t const* pset_end = pset_begin;
        std::advance( pset_end, size_t{ 1 }  );

        this->doSetRequiresCollectFlag( false );

        CpuTrackJob::status_t status = this->doSetParticleSetIndices(
            pset_begin, pset_end, particles_buffer.getCApiPtr() );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = st::TrackJobBaseNew::doReset( particles_buffer.getCApiPtr(),
            beam_elements_buffer.getCApiPtr(), ( ptr_output_buffer != nullptr )
                ? ptr_output_buffer->getCApiPtr() : nullptr,
            until_turn_elem_by_elem );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetCxxBufferPointers(
                particles_buffer, beam_elements_buffer, ptr_output_buffer );
        }
    }

    CpuTrackJob::CpuTrackJob(
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        CpuTrackJob::size_type const particle_set_index,
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CpuTrackJob::size_type const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str ) :
            st::TrackJobBaseNew( st::ARCHITECTURE_CPU,
                SIXTRL_ARCHITECTURE_CPU_STR, config_str )
    {
        using size_t = CpuTrackJob::size_type;

        size_t const* pset_begin = &particle_set_index;
        size_t const* pset_end = pset_begin;
        std::advance( pset_end, size_t{ 1 }  );

        this->doSetRequiresCollectFlag( false );

        CpuTrackJob::status_t status = this->doSetParticleSetIndices(
            pset_begin, pset_end, particles_buffer );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = st::TrackJobBaseNew::doReset( particles_buffer,
            beam_elements_buffer, ptr_output_buffer, until_turn_elem_by_elem );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    CpuTrackJob::CpuTrackJob(
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        CpuTrackJob::size_type const num_particle_sets,
        CpuTrackJob::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        CpuTrackJob::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        CpuTrackJob::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CpuTrackJob::size_type const until_turn_elem_by_elem,
        std::string const& config_str ) :
            st::TrackJobBaseNew( st::ARCHITECTURE_CPU,
                SIXTRL_ARCHITECTURE_CPU_STR, config_str.c_str() )
    {
        using size_t = CpuTrackJob::size_type;
        size_t const* pset_indices_end = pset_indices_begin;
        std::advance( pset_indices_end, num_particle_sets  );

        this->doSetRequiresCollectFlag( false );

        CpuTrackJob::status_t status = this->doSetParticleSetIndices(
            pset_indices_begin, pset_indices_end,
                particles_buffer.getCApiPtr() );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = st::TrackJobBaseNew::doReset( particles_buffer.getCApiPtr(),
            beam_elements_buffer.getCApiPtr(), ( ptr_output_buffer != nullptr )
                ? ptr_output_buffer->getCApiPtr() : nullptr,
                    until_turn_elem_by_elem );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetCxxBufferPointers(
                particles_buffer, beam_elements_buffer, ptr_output_buffer );
        }
    }

    CpuTrackJob::CpuTrackJob(
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        CpuTrackJob::size_type const num_particle_sets,
        CpuTrackJob::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        CpuTrackJob::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        CpuTrackJob::size_type const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str ) :
            st::TrackJobBaseNew( st::ARCHITECTURE_CPU,
                SIXTRL_ARCHITECTURE_CPU_STR, config_str )
    {
        using size_t = CpuTrackJob::size_type;
        size_t const* pset_indices_end = pset_indices_begin;
        std::advance( pset_indices_end, num_particle_sets  );

        this->doSetRequiresCollectFlag( false );

        CpuTrackJob::status_t status = this->doSetParticleSetIndices(
            pset_indices_begin, pset_indices_end, particles_buffer );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );

        status = st::TrackJobBaseNew::doReset( particles_buffer,
            beam_elements_buffer, ptr_output_buffer, until_turn_elem_by_elem );

        SIXTRL_ASSERT( status == st::ARCH_STATUS_SUCCESS );
        ( void )status;
    }

    /* --------------------------------------------------------------------- */

    CpuTrackJob::status_t CpuTrackJob::doFetchParticleAddresses()
    {
        CpuTrackJob::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        CpuTrackJob::buffer_t* ptr_addr_buffer =
            this->doGetPtrParticlesAddrBuffer();

        if( ( !this->isInDebugMode() ) && ( ptr_addr_buffer != nullptr ) )
        {
            status = ::NS(ParticlesAddr_buffer_store_all_addresses)(
                ptr_addr_buffer->getCApiPtr(), this->ptrCParticlesBuffer() );
        }
        else if( ptr_addr_buffer != nullptr )
        {
            st::arch_debugging_t result_register =
                st::ARCH_DEBUGGING_REGISTER_EMPTY;

            status = ::NS(Particles_managed_buffer_store_all_addresses_debug)(
                ::NS(Buffer_get_data_begin)( ptr_addr_buffer->getCApiPtr() ),
                ::NS(Buffer_get_const_data_begin)(
                    this->ptrCParticlesBuffer() ),
                ::NS(Buffer_get_slot_size)( this->ptrCParticlesBuffer() ),
                &result_register );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( ::NS(DebugReg_has_any_flags_set)( result_register ) ) )
            {
                if( ::NS(DebugReg_has_status_flags_set)( result_register ) )
                {
                    status = ::NS(DebugReg_get_stored_arch_status)(
                        result_register );
                }
                else
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                }
            }
        }

        return status;
    }

    CpuTrackJob::track_status_t CpuTrackJob::doTrackUntilTurn(
        CpuTrackJob::size_type const until_turn )
    {
        return st::trackUntil( *this, until_turn );
    }

    CpuTrackJob::track_status_t CpuTrackJob::doTrackElemByElem(
        CpuTrackJob::size_type const until_turn_elem_by_elem )
    {
        return st::trackElemByElem( *this, until_turn_elem_by_elem );
    }

    CpuTrackJob::track_status_t CpuTrackJob::doTrackLine(
        CpuTrackJob::size_type const line_begin_idx,
        CpuTrackJob::size_type const line_end_idx,
        bool const end_turn )
    {
        return st::trackLine( *this, line_begin_idx, line_end_idx, end_turn );
    }
}

#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

/* end: sixtracklib/common/track/track_job_cpu.cpp */
