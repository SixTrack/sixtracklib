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
    #include "sixtracklib/common/track.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/track/track_job_base.hpp"
    #endif /* defined( __cplusplus ) */

#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    CpuTrackJob::track_status_t trackUntilTurn(
        CpuTrackJob& SIXTRL_RESTRICT_REF track_job,
        CpuTrackJob::size_type const until_turn )
    {
        using track_job_t = SIXTRL_CXX_NAMESPACE::CpuTrackJob;
        using size_t      = track_job_t::size_type;
        using particles_t = ::NS(Particles);
        using be_iter_t   = ::NS(Object) const*;

        track_job_t::track_status_t track_status =
            SIXTRL_CXX_NAMESPACE::TRACK_SUCCESS;

        SIXTRL_ASSERT( track_job.ptrCParticlesBuffer() != nullptr );
        SIXTRL_ASSERT( track_job.ptrCBeamElementsBuffer() != nullptr );
        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)(
            track_job.ptrCBeamElementsBuffer() ) > size_t{ 0 } );
        SIXTRL_ASSERT( track_job.numParticleSets() > size_t{ 0 } );

        size_t const* pset_it = track_job.particleSetIndicesBegin();
        size_t const* const pset_end = track_job.particleSetIndicesEnd();

        SIXTRL_ASSERT( pset_it  != nullptr );
        SIXTRL_ASSERT( pset_end != nullptr );

        be_iter_t obj_begin = ::NS(Buffer_get_const_objects_begin)(
            track_job.ptrCBeamElementsBuffer() );

        be_iter_t obj_end = ::NS(Buffer_get_const_objects_end)(
            track_job.ptrCBeamElementsBuffer() );

        if( !track_job.isInDebugMode() )
        {
            while( pset_it != pset_end )
            {
                particles_t* particles = ::NS(Particles_buffer_get_particles)(
                    track_job.ptrCParticlesBuffer(), *pset_it++ );

                SIXTRL_ASSERT( particles != nullptr );

                track_status |= ::NS(Track_all_particles_until_turn_obj)(
                    particles, obj_begin, obj_end, until_turn  );
            }
        }
        else
        {
            while( ( track_status == st::TRACK_SUCCESS ) &&
                   ( pset_it != pset_end ) )
            {
                particles_t* particles = ::NS(Particles_buffer_get_particles)(
                    track_job.ptrCParticlesBuffer(), *pset_it++ );

                SIXTRL_ASSERT( particles != nullptr );

                track_status = ::NS(Track_all_particles_until_turn_obj)(
                    particles, obj_begin, obj_end, until_turn  );
            }
        }

        return track_status;
    }

    CpuTrackJob::track_status_t trackElemByElemUntilTurn(
        CpuTrackJob& SIXTRL_RESTRICT_REF track_job,
        CpuTrackJob::size_type const until_turn_elem_by_elem )
    {
        using track_job_t = SIXTRL_CXX_NAMESPACE::CpuTrackJob;
        using size_t      = track_job_t::size_type;
        using particles_t = ::NS(Particles);
        using be_iter_t   = ::NS(Object) const*;
        using elem_by_elem_config_t = ::NS(ElemByElemConfig);

        track_job_t::track_status_t ret = SIXTRL_CXX_NAMESPACE::TRACK_SUCCESS;
        elem_by_elem_config_t const* config = track_job.ptrElemByElemConfig();

        SIXTRL_ASSERT( track_job.ptrCParticlesBuffer() != nullptr );
        SIXTRL_ASSERT( track_job.ptrCBeamElementsBuffer() != nullptr );
        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)(
            track_job.ptrCBeamElementsBuffer() ) > size_t{ 0 } );
        SIXTRL_ASSERT( track_job.numParticleSets() > size_t{ 0 } );
        SIXTRL_ASSERT( track_job.hasOutputBuffer() );
        SIXTRL_ASSERT( track_job.hasElemByElemOutput() );
        SIXTRL_ASSERT( track_job.hasElemByElemConfig() );
        SIXTRL_ASSERT( config != nullptr );

        SIXTRL_ASSERT( ::NS(ElemByElemConfig_get_output_store_address)(
            config ) != ::NS(elem_by_elem_out_addr_t){ 0 } );

        size_t const* pset_it  = track_job.particleSetIndicesBegin();
        size_t const* const pset_end = track_job.particleSetIndicesEnd();

        SIXTRL_ASSERT( pset_it  != nullptr );
        SIXTRL_ASSERT( pset_end != nullptr );

        be_iter_t obj_begin = ::NS(Buffer_get_const_objects_begin)(
            track_job.ptrCBeamElementsBuffer() );

        be_iter_t obj_end = ::NS(Buffer_get_const_objects_end)(
            track_job.ptrCBeamElementsBuffer() );

        SIXTRL_ASSERT( obj_begin != nullptr );
        SIXTRL_ASSERT( obj_end != nullptr );

        if( !track_job.isInDebugMode() )
        {
            while( pset_it != pset_end )
            {
                particles_t* particles = ::NS(Particles_buffer_get_particles)(
                    track_job.ptrCParticlesBuffer(), *pset_it++ );

                SIXTRL_ASSERT( particles != nullptr );

                ret |= ::NS(Track_all_particles_element_by_element_until_turn_objs)(
                    particles, config, obj_begin, obj_end,
                        until_turn_elem_by_elem );
            }
        }
        else
        {
            while( ( ret == SIXTRL_CXX_NAMESPACE::TRACK_SUCCESS ) &&
                   ( pset_it != pset_end ) )
            {
                particles_t* particles = ::NS(Particles_buffer_get_particles)(
                    track_job.ptrCParticlesBuffer(), *pset_it++ );

                SIXTRL_ASSERT( particles != nullptr );

                ret = ::NS(Track_all_particles_element_by_element_until_turn_objs)(
                    particles, config, obj_begin, obj_end, until_turn_elem_by_elem );
            }
        }

        return ret;
    }

    CpuTrackJob::track_status_t trackLine(
        CpuTrackJob& SIXTRL_RESTRICT_REF track_job,
        CpuTrackJob::size_type const belem_begin_idx,
        CpuTrackJob::size_type const belem_end_idx,
        bool const finish_turn )
    {
        using track_job_t = SIXTRL_CXX_NAMESPACE::CpuTrackJob;
        using size_t      = track_job_t::size_type;
        using particles_t = ::NS(Particles);

        track_job_t::track_status_t ret = SIXTRL_CXX_NAMESPACE::TRACK_SUCCESS;

        SIXTRL_ASSERT( track_job.ptrCParticlesBuffer() != nullptr );
        SIXTRL_ASSERT( track_job.ptrCBeamElementsBuffer() != nullptr );
        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)(
            track_job.ptrCBeamElementsBuffer() ) > size_t{ 0 } );
        SIXTRL_ASSERT( track_job.numParticleSets() > size_t{ 0 } );

        size_t const* pset_it  = track_job.particleSetIndicesBegin();
        size_t const* const pset_end = track_job.particleSetIndicesEnd();

        SIXTRL_ASSERT( pset_it  != nullptr );
        SIXTRL_ASSERT( pset_end != nullptr );
        SIXTRL_ASSERT( belem_begin_idx <= belem_end_idx );
        SIXTRL_ASSERT( belem_end_idx <= ::NS(Buffer_get_num_of_objects)(
            track_job.ptrCBeamElementsBuffer() ) );

        if( !track_job.isInDebugMode() )
        {
            while( pset_it != pset_end )
            {
                particles_t* particles = ::NS(Particles_buffer_get_particles)(
                    track_job.ptrCParticlesBuffer(), *pset_it++ );

                SIXTRL_ASSERT( particles != nullptr );

                ret |= ::NS(Track_all_particles_line_ext)(
                    particles, track_job.ptrCBeamElementsBuffer(),
                        belem_begin_idx, belem_end_idx, finish_turn );
            }
        }
        else
        {
            while( ( ret == SIXTRL_CXX_NAMESPACE::TRACK_SUCCESS ) &&
                   ( pset_it != pset_end ) )
            {
                particles_t* particles = ::NS(Particles_buffer_get_particles)(
                    track_job.ptrCParticlesBuffer(), *pset_it++ );

                SIXTRL_ASSERT( particles != nullptr );

                ret = ::NS(Track_all_particles_line_ext)(
                    particles, track_job.ptrCBeamElementsBuffer(), belem_begin_idx,
                        belem_end_idx, finish_turn );
            }
        }

        return ret;
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
        return st::trackUntilTurn( *this, until_turn );
    }

    CpuTrackJob::track_status_t CpuTrackJob::doTrackElemByElem(
        CpuTrackJob::size_type const until_turn_elem_by_elem )
    {
        return st::trackElemByElemUntilTurn( *this, until_turn_elem_by_elem );
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
