#include "sixtracklib/common/track/track_job_base.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
        #include <string>
        #include <vector>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/be_monitor/output_buffer.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/particles/definitions.h"
    #include "sixtracklib/common/particles/particles_addr.h"
    #include "sixtracklib/common/particles/particles_addr.hpp"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/control/node_id.hpp"
    #include "sixtracklib/common/control/arch_info.hpp"
    #include "sixtracklib/common/control/arch_base.hpp"
    #include "sixtracklib/common/track/definitions.h"

    #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
        SIXTRACKLIB_ENABLE_MODULE_CUDA == 1

    #include "sixtracklib/cuda/track_job.hpp"

    #endif /* CUDA */


#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace st = SIXTRL_CXX_NAMESPACE;
namespace SIXTRL_CXX_NAMESPACE
{
    using _this_t = st::TrackJobBaseNew;

    _this_t::size_type
    TrackJobBaseNew::DefaultNumParticleSetIndices() SIXTRL_NOEXCEPT
    {
        return st::TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS;
    }

    _this_t::size_type const*
    TrackJobBaseNew::DefaultParticleSetIndicesBegin() SIXTRL_NOEXCEPT
    {
        return &st::TRACK_JOB_DEFAULT_PARTICLE_SET_INDICES[ 0 ];
    }

    _this_t::size_type const*
    TrackJobBaseNew::DefaultParticleSetIndicesEnd() SIXTRL_NOEXCEPT
    {
        _this_t::size_type const* end_ptr =
            TrackJobBaseNew::DefaultParticleSetIndicesBegin();

        std::advance( end_ptr, st::TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS );
        return end_ptr;
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::clear()
    {
        this->doClear( this->doGetDefaultAllClearFlags() );
    }

    /* --------------------------------------------------------------------- */

    _this_t::collect_flag_t TrackJobBaseNew::collect()
    {
        return this->doCollect( this->m_collect_flags );
    }

    _this_t::collect_flag_t TrackJobBaseNew::collect(
        _this_t::collect_flag_t const flags )
    {
        return this->doCollect( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    _this_t::status_t TrackJobBaseNew::collectParticles()
    {
        return ( st::TrackJobBaseNew::IsCollectFlagSet(
            this->doCollect( st::TRACK_JOB_IO_PARTICLES ),
                st::TRACK_JOB_IO_PARTICLES ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t TrackJobBaseNew::collectBeamElements()
    {
        return ( st::TrackJobBaseNew::IsCollectFlagSet(
            this->doCollect( st::TRACK_JOB_IO_BEAM_ELEMENTS ),
                st::TRACK_JOB_IO_BEAM_ELEMENTS ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t TrackJobBaseNew::collectOutput()
    {
        return ( TrackJobBaseNew::IsCollectFlagSet(
            this->doCollect( st::TRACK_JOB_IO_OUTPUT ),
                st::TRACK_JOB_IO_OUTPUT ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t TrackJobBaseNew::collectDebugFlag()
    {
        return ( TrackJobBaseNew::IsCollectFlagSet(
            this->doCollect( st::TRACK_JOB_IO_DEBUG_REGISTER ),
                st::TRACK_JOB_IO_DEBUG_REGISTER ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t TrackJobBaseNew::collectParticlesAddresses()
    {
        return ( TrackJobBaseNew::IsCollectFlagSet(
            this->doCollect( st::TRACK_JOB_IO_PARTICLES_ADDR ),
                st::TRACK_JOB_IO_PARTICLES_ADDR ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    void TrackJobBaseNew::enableCollectParticles()  SIXTRL_NOEXCEPT
    {
        this->m_collect_flags |= st::TRACK_JOB_IO_PARTICLES;
    }

    void TrackJobBaseNew::disableCollectParticles() SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = TrackJobBaseNew::UnsetCollectFlag(
            this->m_collect_flags, st::TRACK_JOB_IO_PARTICLES );
    }

    bool TrackJobBaseNew::isCollectingParticles() const SIXTRL_NOEXCEPT
    {
        return TrackJobBaseNew::IsCollectFlagSet( this->m_collect_flags,
            st::TRACK_JOB_IO_PARTICLES );
    }

    void TrackJobBaseNew::enableCollectBeamElements()  SIXTRL_NOEXCEPT
    {
        this->m_collect_flags |= st::TRACK_JOB_IO_BEAM_ELEMENTS;
    }

    void TrackJobBaseNew::disableCollectBeamElements() SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = TrackJobBaseNew::UnsetCollectFlag(
            this->m_collect_flags, st::TRACK_JOB_IO_BEAM_ELEMENTS );
    }

    bool TrackJobBaseNew::isCollectingBeamElements() const SIXTRL_NOEXCEPT
    {
        return TrackJobBaseNew::IsCollectFlagSet( this->m_collect_flags,
                st::TRACK_JOB_IO_BEAM_ELEMENTS );
    }

    void TrackJobBaseNew::enableCollectOutput()  SIXTRL_NOEXCEPT
    {
        this->m_collect_flags |= st::TRACK_JOB_IO_OUTPUT;
    }

    void TrackJobBaseNew::disableCollectOutput() SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = TrackJobBaseNew::UnsetCollectFlag(
            this->m_collect_flags, st::TRACK_JOB_IO_OUTPUT );
    }

    bool TrackJobBaseNew::isCollectingOutput() const SIXTRL_NOEXCEPT
    {
        return TrackJobBaseNew::IsCollectFlagSet( this->m_collect_flags,
                st::TRACK_JOB_IO_OUTPUT );
    }

    _this_t::collect_flag_t
    TrackJobBaseNew::collectFlags() const SIXTRL_NOEXCEPT
    {
        return this->m_collect_flags;
    }

    void TrackJobBaseNew::setCollectFlags(
        _this_t::collect_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = ( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    bool TrackJobBaseNew::requiresCollecting() const SIXTRL_NOEXCEPT
    {
        return this->m_requires_collect;
    }

    /* --------------------------------------------------------------------- */

    _this_t::push_flag_t TrackJobBaseNew::push(
        _this_t::push_flag_t const push_flag )
    {
        return this->doPush( push_flag & st::TRACK_JOB_PUSH_ALL );
    }

    _this_t::status_t TrackJobBaseNew::pushParticles()
    {
        return ( ( st::TRACK_JOB_IO_PARTICLES & this->push(
            st::TRACK_JOB_IO_PARTICLES ) ) == st::TRACK_JOB_IO_PARTICLES )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t TrackJobBaseNew::pushBeamElements()
    {
        return ( ( st::TRACK_JOB_IO_BEAM_ELEMENTS & this->push(
            st::TRACK_JOB_IO_BEAM_ELEMENTS ) ) == st::TRACK_JOB_IO_BEAM_ELEMENTS )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t TrackJobBaseNew::pushOutput()
    {
        return ( ( st::TRACK_JOB_IO_OUTPUT & this->push(
            st::TRACK_JOB_IO_OUTPUT ) ) == st::TRACK_JOB_IO_OUTPUT )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    _this_t::status_t TrackJobBaseNew::fetchParticleAddresses()
    {
        _this_t::status_t status = this->doFetchParticleAddresses();

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( this->requiresCollecting() ) )
        {
            status = this->collectParticlesAddresses();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetHasParticleAddressesFlag( true );
        }

        return status;
    }

    _this_t::status_t TrackJobBaseNew::clearParticleAddresses(
        _this_t::size_type const index )
    {
        return this->doClearParticleAddresses( index );
    }

    _this_t::status_t TrackJobBaseNew::clearAllParticleAddresses()
    {
        return this->doClearAllParticleAddresses();
    }

    bool TrackJobBaseNew::canFetchParticleAddresses() const SIXTRL_NOEXCEPT
    {
        return ( this->doGetPtrParticlesAddrBuffer() != nullptr );
    }

    bool TrackJobBaseNew::hasParticleAddresses() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( !this->m_has_particle_addresses ) ||
            ( ( this->m_has_particle_addresses ) &&
              ( this->canFetchParticleAddresses() ) ) );

        return this->m_has_particle_addresses;
    }

    TrackJobBaseNew::particles_addr_t const*
    TrackJobBaseNew::particleAddresses(
        _this_t::size_type const index ) const SIXTRL_NOEXCEPT
    {
        using ptr_paddr_t = TrackJobBaseNew::particles_addr_t const*;
        ptr_paddr_t ptr_paddr = nullptr;

        if( ( this->doGetPtrParticlesAddrBuffer() != nullptr ) &&
            ( this->doGetPtrParticlesAddrBuffer()->getNumObjects() > index ) &&
            ( this->hasParticleAddresses() ) )
        {
            ptr_paddr = ::NS(ParticlesAddr_buffer_get_const_particle_addr)(
                this->doGetPtrParticlesAddrBuffer()->getCApiPtr(), index );
        }

        return ptr_paddr;
    }

    _this_t::buffer_t const*
    TrackJobBaseNew::ptrParticleAddressesBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrParticlesAddrBuffer();
    }

    _this_t::c_buffer_t const*
    TrackJobBaseNew::ptrCParticleAddressesBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->doGetPtrParticlesAddrBuffer() != nullptr )
            ? this->doGetPtrParticlesAddrBuffer()->getCApiPtr() : nullptr;
    }

    /* --------------------------------------------------------------------- */

    TrackJobBaseNew::track_status_t TrackJobBaseNew::trackUntil(
        _this_t::size_type const until_turn )
    {
        return this->doTrackUntilTurn( until_turn );
    }

    TrackJobBaseNew::track_status_t TrackJobBaseNew::trackElemByElem(
        _this_t::size_type const until_turn_elem_by_elem )
    {
        return this->doTrackElemByElem( until_turn_elem_by_elem );
    }

    TrackJobBaseNew::track_status_t TrackJobBaseNew::trackLine(
        _this_t::size_type const be_begin_index,
        _this_t::size_type const be_end_index,
        bool const finish_turn )
    {
        return this->doTrackLine( be_begin_index, be_end_index, finish_turn );
    }

    /* --------------------------------------------------------------------- */

    _this_t::status_t TrackJobBaseNew::reset(
        _this_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        _this_t::buffer_t& SIXTRL_RESTRICT_REF be_buffer,
        _this_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem  )
    {
        using c_buffer_t = _this_t::c_buffer_t;
        using clear_flag_t = TrackJobBaseNew::clear_flag_t;

        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        c_buffer_t* ptr_pb  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_eb  = be_buffer.getCApiPtr();
        c_buffer_t* ptr_out = ( ptr_output_buffer != nullptr ) ?
            ptr_output_buffer->getCApiPtr() : nullptr;

        clear_flag_t const clear_flags = this->doPrepareResetClearFlags( ptr_pb,
            this->numParticleSets(), this->particleSetIndicesBegin(),
                ptr_eb, ptr_out, until_turn_elem_by_elem );

        this->doClear( clear_flags );

        status = this->doReset( ptr_pb, ptr_eb, ptr_out,
            until_turn_elem_by_elem );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetCxxBufferPointers(
                particles_buffer, be_buffer, ptr_output_buffer );
        }

        return status;
    }

    _this_t::status_t TrackJobBaseNew::reset(
        _this_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        _this_t::size_type const pset_index,
        _this_t::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        _this_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _this_t::size_type const until_turn_elem_by_elem  )
    {
        _this_t::c_buffer_t* ptr_out = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        _this_t::status_t status = TrackJobBaseNew::reset(
            particles_buffer.getCApiPtr(), size_t{ 1 }, &pset_index,
                beam_elements_buffer.getCApiPtr(), ptr_out,
                    until_turn_elem_by_elem );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetCxxBufferPointers(
                particles_buffer, beam_elements_buffer, ptr_output_buffer );
        }

        return status;
    }

    _this_t::status_t TrackJobBaseNew::reset(
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _this_t::size_type const until_turn_elem_by_elem  )
    {
        using clear_flag_t = TrackJobBaseNew::clear_flag_t;

        clear_flag_t const clear_flags = this->doPrepareResetClearFlags(
            particles_buffer, this->numParticleSets(),
                this->particleSetIndicesBegin(), be_buffer, ptr_output_buffer,
                    until_turn_elem_by_elem );

        this->doClear( clear_flags );

        return this->doReset( particles_buffer, be_buffer, ptr_output_buffer,
            until_turn_elem_by_elem );
    }

    _this_t::status_t TrackJobBaseNew::reset(
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _this_t::size_type const particle_set_index,
        _this_t::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _this_t::size_type const until_turn_elem_by_elem  )
    {
        return TrackJobBaseNew::reset( particles_buffer,
            _this_t::size_type{ 1 }, &particle_set_index, be_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
    }

    _this_t::status_t TrackJobBaseNew::reset(
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _this_t::size_type const num_particle_sets,
        _this_t::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        _this_t::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _this_t::size_type const until_turn_elem_by_elem  )
    {
        using size_t = _this_t::size_type;

        TrackJobBaseNew::clear_flag_t const clear_flags =
            this->doPrepareResetClearFlags( particles_buffer,
                num_particle_sets, pset_indices_begin, beam_elements_buffer,
                    ptr_output_buffer, until_turn_elem_by_elem );

        this->doClear( clear_flags );

        _this_t::status_t status = st::ARCH_STATUS_SUCCESS;

        if( ( pset_indices_begin != nullptr ) &&
            ( num_particle_sets > size_t{ 0 } ) )
        {
            status = this->doSetParticleSetIndices( pset_indices_begin,
                pset_indices_begin + num_particle_sets, particles_buffer );
        }
        else
        {
            this->doInitDefaultParticleSetIndices();
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doReset( particles_buffer, beam_elements_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    /*
    _this_t::status_t TrackJobBaseNew::selectParticleSets(
        _this_t::size_type const num_particle_sets,
        _this_t::size_type const*
            SIXTRL_RESTRICT particle_set_indices_begin )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->ptrCParticlesBuffer() != nullptr )
        {


        }

        return status;
    }
    */

    _this_t::status_t TrackJobBaseNew::selectParticleSet(
        _this_t::size_type const particle_set_index )
    {
        using buffer_t   = _this_t::buffer_t;
        using c_buffer_t = _this_t::c_buffer_t;
        using size_t = _this_t::size_type;

        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        buffer_t*   ptr_particles_buffer   = this->ptrOutputBuffer();
        buffer_t*   ptr_beam_elem_buffer   = this->ptrBeamElementsBuffer();

        c_buffer_t* ptr_c_particles_buffer = this->ptrCParticlesBuffer();
        c_buffer_t* ptr_c_beam_elem_buffer = this->ptrCBeamElementsBuffer();

        if( ( ptr_c_particles_buffer != nullptr ) &&
            ( !::NS(Buffer_needs_remapping)( ptr_c_particles_buffer ) ) &&
            ( static_cast< size_t >( ::NS(Buffer_get_num_of_objects)(
                ptr_c_particles_buffer ) ) > particle_set_index ) &&
            ( ptr_c_beam_elem_buffer != nullptr ) &&
            ( !::NS(Buffer_needs_remapping)( ptr_c_beam_elem_buffer ) ) )
        {
            buffer_t* ptr_output_buffer = nullptr;
            c_buffer_t* ptr_c_output_buffer = nullptr;

            if( ( this->hasOutputBuffer() ) && ( !this->ownsOutputBuffer() ) )
            {
                ptr_output_buffer   = this->ptrOutputBuffer();
                ptr_c_output_buffer = this->ptrCOutputBuffer();

                SIXTRL_ASSERT( ::NS(Buffer_needs_remapping)(
                    ptr_c_output_buffer ) );
            }

            if( ( ptr_particles_buffer != nullptr ) &&
                ( ptr_beam_elem_buffer != nullptr ) )
            {
                SIXTRL_ASSERT(
                    ( ( ptr_output_buffer != nullptr ) &&
                      ( ptr_c_output_buffer != nullptr ) ) ||
                    ( ( ptr_output_buffer == nullptr ) &&
                      ( ptr_c_output_buffer == nullptr ) ) );

                SIXTRL_ASSERT(
                    ( ptr_particles_buffer->getCApiPtr() ==
                      ptr_c_particles_buffer ) &&
                    ( ptr_beam_elem_buffer->getCApiPtr() ==
                      ptr_c_beam_elem_buffer ) );

                if( ptr_c_output_buffer != nullptr )
                {
                    ::NS(Buffer_clear)( ptr_c_output_buffer, true );
                }

                size_t particle_set_indices[ 1 ] = { size_t{ 0 } };
                particle_set_indices[ 0 ] = particle_set_index;

                status = this->reset( *ptr_particles_buffer,
                    &particle_set_indices[ 0 ], &particle_set_indices[ 1 ],
                        *ptr_beam_elem_buffer, ptr_output_buffer,
                            this->untilTurnElemByElem() );
            }
            else if( ( ptr_c_particles_buffer != nullptr ) &&
                     ( ptr_c_beam_elem_buffer != nullptr ) )
            {
                if( ptr_c_output_buffer != nullptr )
                {
                    ::NS(Buffer_clear)( ptr_c_output_buffer, true );
                }

                status = this->reset( ptr_c_particles_buffer, size_t{ 1 },
                    &particle_set_index, ptr_c_beam_elem_buffer,
                        ptr_c_output_buffer, this->untilTurnElemByElem() );
            }
        }

        return status;
    }



    /* --------------------------------------------------------------------- */

    _this_t::status_t TrackJobBaseNew::assignOutputBuffer(
        _this_t::buffer_t& SIXTRL_RESTRICT_REF output_buffer )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->doAssignNewOutputBuffer( output_buffer.getCApiPtr() ) )
        {
            if( this->hasOutputBuffer() )
            {
                SIXTRL_ASSERT( output_buffer.getCApiPtr() ==
                    this->ptrCOutputBuffer() );

                this->doSetPtrOutputBuffer( &output_buffer );
            }

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    _this_t::status_t TrackJobBaseNew::assignOutputBuffer(
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        return this->doAssignNewOutputBuffer( ptr_output_buffer );
    }

    /* --------------------------------------------------------------------- */

    _this_t::size_type TrackJobBaseNew::numParticleSets() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_indices.size();
    }

    _this_t::size_type const*
    TrackJobBaseNew::particleSetIndicesBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_indices.data();
    }

    _this_t::size_type const*
    TrackJobBaseNew::particleSetIndicesEnd() const SIXTRL_NOEXCEPT
    {
        _this_t::size_type const* ptr =
            this->particleSetIndicesBegin();

        SIXTRL_ASSERT( ptr != nullptr );
        std::advance( ptr, this->numParticleSets() );
        return ptr;
    }

    _this_t::size_type TrackJobBaseNew::particleSetIndex(
        _this_t::size_type const idx ) const
    {
        return this->m_particle_set_indices.at( idx );
    }

    /* --------------------------------------------------------------------- */

    _this_t::num_particles_t TrackJobBaseNew::particleSetBeginIndexOffset(
        _this_t::size_type const pset_index ) const
    {
        return this->m_particle_set_begin_offsets.at( pset_index );
    }

    _this_t::num_particles_t TrackJobBaseNew::particleSetEndIndexOffset(
        _this_t::size_type const pset_index ) const
    {
        return this->m_particle_set_end_offsets.at( pset_index );
    }

    _this_t::num_particles_t TrackJobBaseNew::numParticlesInSet(
        _this_t::size_type const pset_index ) const SIXTRL_NOEXCEPT
    {
        using num_particles_t = _this_t::num_particles_t;
        num_particles_t num_particles_in_set = num_particles_t{ 0 };

        if( ( pset_index < this->numParticleSets() ) &&
            ( this->m_particle_set_begin_offsets.size() ==
              this->m_particle_set_end_offsets.size() ) &&
            ( this->m_particle_set_begin_offsets.size() ==
              this->numParticleSets() ) )
        {
            num_particles_t const begin_offset =
                this->m_particle_set_begin_offsets[ pset_index ];

            num_particles_t const end_offset =
                this->m_particle_set_end_offsets[ pset_index ];

            if( begin_offset < end_offset )
            {
                num_particles_in_set = end_offset - begin_offset;
            }
        }

        return num_particles_in_set;
    }

    _this_t::num_particles_t const*
    TrackJobBaseNew::particleSetBeginOffsetsBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_begin_offsets.data();
    }

    _this_t::num_particles_t const*
    TrackJobBaseNew::particleSetBeginOffsetsEnd() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->numParticleSets() ==
                       this->m_particle_set_begin_offsets.size() );

        _this_t::num_particles_t const* end_ptr =
            this->particleSetBeginOffsetsBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numParticleSets() );
        }

        return end_ptr;
    }

    _this_t::num_particles_t const*
    TrackJobBaseNew::particleSetEndOffsetsBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_end_offsets.data();
    }

    _this_t::num_particles_t const*
    TrackJobBaseNew::particleSetEndOffsetsEnd() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->numParticleSets() ==
                       this->m_particle_set_end_offsets.size() );

        _this_t::num_particles_t const* end_ptr =
            this->particleSetEndOffsetsBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numParticleSets() );
        }

        return end_ptr;
    }

    _this_t::num_particles_t
    TrackJobBaseNew::totalNumParticles() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles;
    }

    _this_t::num_particles_t
    TrackJobBaseNew::totalNumParticlesInParticleSets() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles_in_sets;
    }

    _this_t::size_type
    TrackJobBaseNew::totalNumOfAvailableParticleSets() const SIXTRL_NOEXCEPT
    {
        return this->m_num_particle_sets_in_buffer;
    }

    /* --------------------------------------------------------------------- */

    _this_t::particle_index_t
    TrackJobBaseNew::minParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_particle_id;
    }

    _this_t::particle_index_t
    TrackJobBaseNew::maxParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_particle_id;
    }

    _this_t::particle_index_t
    TrackJobBaseNew::minElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_min_element_id;
    }

    _this_t::particle_index_t
    TrackJobBaseNew::maxElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_max_element_id;
    }

    _this_t::particle_index_t
    TrackJobBaseNew::minInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_initial_turn_id;
    }

    _this_t::particle_index_t
    TrackJobBaseNew::maxInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_initial_turn_id;
    }

    /* --------------------------------------------------------------------- */

    _this_t::size_type
    TrackJobBaseNew::totalNumOfAvailableBeamElements() const SIXTRL_NOEXCEPT
    {
        return this->m_num_beam_elements_in_buffer;
    }

    /* --------------------------------------------------------------------- */

    _this_t::buffer_t*
    TrackJobBaseNew::ptrParticlesBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::_this_t::buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrParticlesBuffer() );
    }

    _this_t::buffer_t const*
    TrackJobBaseNew::ptrParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_particles_buffer;
    }

    _this_t::c_buffer_t*
    TrackJobBaseNew::ptrCParticlesBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::_this_t::c_buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrCParticlesBuffer() );
    }

    _this_t::c_buffer_t const*
    TrackJobBaseNew::ptrCParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_c_particles_buffer;
    }

    /* --------------------------------------------------------------------- */

    _this_t::buffer_t*
    TrackJobBaseNew::ptrBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::_this_t::buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrBeamElementsBuffer() );
    }

    _this_t::buffer_t const*
    TrackJobBaseNew::ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_beam_elem_buffer;
    }

    _this_t::c_buffer_t*
    TrackJobBaseNew::ptrCBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::_this_t::c_buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrCBeamElementsBuffer() );
    }

    _this_t::c_buffer_t const*
    TrackJobBaseNew::ptrCBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_c_beam_elem_buffer;
    }

    /* ---------------------------------------------------------------- */

    bool TrackJobBaseNew::hasOutputBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCOutputBuffer() != nullptr );
    }

    bool TrackJobBaseNew::ownsOutputBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_my_output_buffer.get() == nullptr ) ||
            ( ( this->m_my_output_buffer.get() ==
                this->m_ptr_output_buffer ) &&
              ( this->m_my_output_buffer->getCApiPtr() ==
                this->m_ptr_c_output_buffer ) ) );

        return ( ( this->ptrOutputBuffer() != nullptr ) &&
                 ( this->m_my_output_buffer.get() != nullptr ) );
    }

    bool TrackJobBaseNew::hasElemByElemOutput() const SIXTRL_NOEXCEPT
    {
        return this->m_has_elem_by_elem_output;
    }

    bool TrackJobBaseNew::hasBeamMonitorOutput() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( (  !this->m_has_beam_monitor_output ) ||
            ( ( this->m_has_beam_monitor_output ) &&
              ( this->m_ptr_c_output_buffer != nullptr ) ) );

        return this->m_has_beam_monitor_output;
    }

    _this_t::size_type
    TrackJobBaseNew::beamMonitorsOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_be_mon_output_buffer_offset ) ) ||
            ( this->m_be_mon_output_buffer_offset ==
                _this_t::size_type{ 0 } ) );

        return this->m_be_mon_output_buffer_offset;
    }

    _this_t::size_type
    TrackJobBaseNew::elemByElemOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_elem_by_elem_output_offset ) ) ||
            ( this->m_elem_by_elem_output_offset ==
                _this_t::size_type{ 0 } ) );

        return this->m_elem_by_elem_output_offset;
    }

    _this_t::particle_index_t
    TrackJobBaseNew::untilTurnElemByElem() const SIXTRL_NOEXCEPT
    {
        return this->m_until_turn_elem_by_elem;
    }

    _this_t::size_type
    TrackJobBaseNew::numElemByElemTurns() const SIXTRL_NOEXCEPT
    {
        if( ( this->m_until_turn_elem_by_elem > this->m_min_initial_turn_id ) &&
            ( this->m_min_initial_turn_id >= _this_t::particle_index_t{ 0 } ) )
        {
            return static_cast< size_t >( this->m_until_turn_elem_by_elem -
                this->m_min_initial_turn_id );
        }

        return _this_t::size_type{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    _this_t::buffer_t* TrackJobBaseNew::ptrOutputBuffer() SIXTRL_RESTRICT
    {
        return const_cast< st::_this_t::buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrOutputBuffer() );
    }

    _this_t::buffer_t const*
    TrackJobBaseNew::ptrOutputBuffer() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( ( this->m_ptr_output_buffer == nullptr ) ||
            ( this->m_ptr_output_buffer->getCApiPtr() ==
              this->m_ptr_c_output_buffer ) );

        return this->m_ptr_output_buffer;
    }

    _this_t::c_buffer_t* TrackJobBaseNew::ptrCOutputBuffer() SIXTRL_RESTRICT
    {
        return const_cast< st::_this_t::c_buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrCOutputBuffer() );
    }

    _this_t::c_buffer_t const*
    TrackJobBaseNew::ptrCOutputBuffer() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT( ( this->m_ptr_output_buffer == nullptr ) ||
            ( this->m_ptr_output_buffer->getCApiPtr() ==
              this->m_ptr_c_output_buffer ) );

        return this->m_ptr_c_output_buffer;
    }

    /* --------------------------------------------------------------------- */

    bool TrackJobBaseNew::hasBeamMonitors() const SIXTRL_NOEXCEPT
    {
        return !this->m_beam_monitor_indices.empty();
    }

    _this_t::size_type TrackJobBaseNew::numBeamMonitors() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_indices.size();
    }

    _this_t::size_type const*
    TrackJobBaseNew::beamMonitorIndicesBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_indices.data();
    }

    _this_t::size_type const*
    TrackJobBaseNew::beamMonitorIndicesEnd() const SIXTRL_NOEXCEPT
    {
        _this_t::size_type const* ptr = this->beamMonitorIndicesBegin();
        if( ptr != nullptr ) std::advance( ptr, this->numBeamMonitors() );
        return ptr;
    }

    _this_t::size_type TrackJobBaseNew::beamMonitorIndex(
        _this_t::size_type const idx ) const
    {
        return this->m_beam_monitor_indices.at( idx );
    }

    /* --------------------------------------------------------------------- */

    bool TrackJobBaseNew::hasElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_my_elem_by_elem_config.get() == nullptr ) ||
            ( this->m_ptr_c_output_buffer != nullptr ) );

        return ( ::NS(ElemByElemConfig_is_active)(
            this->m_my_elem_by_elem_config.get() ) );
    }

    _this_t::elem_by_elem_config_t const*
    TrackJobBaseNew::ptrElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        return this->m_my_elem_by_elem_config.get();
    }

    _this_t::elem_by_elem_config_t*
    TrackJobBaseNew::ptrElemByElemConfig() SIXTRL_NOEXCEPT
    {
        return const_cast< st::TrackJobBaseNew::elem_by_elem_config_t* >(
            static_cast< st::TrackJobBaseNew const& >(
                *this ).ptrElemByElemConfig() );
    }

    bool TrackJobBaseNew::elemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return NS(ElemByElemConfig_is_rolling)( this->ptrElemByElemConfig() );
    }

    bool TrackJobBaseNew::defaultElemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return this->m_default_elem_by_elem_rolling;
    }

    void TrackJobBaseNew::setDefaultElemByElemRolling(
        bool const is_rolling ) SIXTRL_NOEXCEPT
    {
        this->m_default_elem_by_elem_rolling = is_rolling;
    }

    /* --------------------------------------------------------------------- */

    _this_t::clear_flag_t
    TrackJobBaseNew::doGetDefaultAllClearFlags() const SIXTRL_NOEXCEPT
    {
        return this->m_clear_all_flags;
    }

    void TrackJobBaseNew::doSetDefaultAllClearFlags(
        _this_t::clear_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        this->m_clear_all_flags = flags;
    }

    _this_t::clear_flag_t
    TrackJobBaseNew::doGetDefaultPrepareResetClearFlags() const SIXTRL_NOEXCEPT
    {
        return this->m_clear_prepare_reset_flags;
    }

    void TrackJobBaseNew::doSetDefaultPrepareResetClearFlags(
        _this_t::clear_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        this->m_clear_prepare_reset_flags = flags;
    }

    /* --------------------------------------------------------------------- */

    _this_t::elem_by_elem_order_t
    TrackJobBaseNew::elemByElemOrder() const SIXTRL_NOEXCEPT
    {
        return ::NS(ElemByElemConfig_get_order)(
            this->m_my_elem_by_elem_config.get() );
    }

    _this_t::elem_by_elem_order_t
    TrackJobBaseNew::defaultElemByElemOrder() const SIXTRL_NOEXCEPT
    {
        return this->m_default_elem_by_elem_order;
    }

    void TrackJobBaseNew::setDefaultElemByElemOrder(
        _this_t::elem_by_elem_order_t const order ) SIXTRL_NOEXCEPT
    {
        this->m_default_elem_by_elem_order = order;
    }

    /* --------------------------------------------------------------------- */

    TrackJobBaseNew::TrackJobBaseNew(
        _this_t::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str ) :
        st::ArchDebugBase( arch_id, arch_str, config_str ),
        m_particle_set_indices(),
        m_particle_set_begin_offsets(),
        m_particle_set_end_offsets(),
        m_beam_monitor_indices(),
        m_my_output_buffer( nullptr ),
        m_my_particles_addr_buffer( new _this_t::buffer_t ),
        m_my_elem_by_elem_config( nullptr ),
        m_ptr_particles_buffer( nullptr ),
        m_ptr_beam_elem_buffer( nullptr ), m_ptr_output_buffer( nullptr ),
        m_ptr_c_particles_buffer( nullptr ),
        m_ptr_c_beam_elem_buffer( nullptr ), m_ptr_c_output_buffer( nullptr ),
        m_be_mon_output_buffer_offset( _this_t::size_type{ 0 } ),
        m_elem_by_elem_output_offset( _this_t::size_type{ 0 } ),
        m_num_particle_sets_in_buffer( _this_t::size_type{ 0 } ),
        m_num_beam_elements_in_buffer( _this_t::size_type{ 0 } ),
        m_default_elem_by_elem_order( ::NS(ELEM_BY_ELEM_ORDER_DEFAULT) ),
        m_total_num_particles( _this_t::num_particles_t{ 0 } ),
        m_total_num_particles_in_sets( _this_t::num_particles_t{ 0 } ),
        m_min_particle_id( _this_t::particle_index_t{ 0 } ),
        m_max_particle_id( _this_t::particle_index_t{ 0 } ),
        m_min_element_id( _this_t::particle_index_t{ 0 } ),
        m_max_element_id( _this_t::particle_index_t{ 0 } ),
        m_min_initial_turn_id( _this_t::particle_index_t{ 0 } ),
        m_max_initial_turn_id( _this_t::particle_index_t{ 0 } ),
        m_until_turn_elem_by_elem( _this_t::size_type{ 0 } ),
        m_collect_flags( st::TRACK_JOB_IO_DEFAULT_FLAGS ),
        m_clear_prepare_reset_flags( st::TRACK_JOB_DEFAULT_CLEAR_FLAGS ),
        m_clear_all_flags( st::TRACK_JOB_CLEAR_ALL_FLAGS ),
        m_default_elem_by_elem_rolling( true ),
        m_has_beam_monitor_output( false ), m_has_elem_by_elem_output( false ),
        m_has_particle_addresses( false ), m_requires_collect( true ),
        m_uses_controller( false ), m_uses_arguments( false )
    {
        this->doInitDefaultParticleSetIndices();
        this->doInitDefaultBeamMonitorIndices();
    }

    TrackJobBaseNew::TrackJobBaseNew( TrackJobBaseNew const& other ) :
        st::ArchDebugBase( other ),
        m_particle_set_indices( other.m_particle_set_indices ),
        m_particle_set_begin_offsets( other.m_particle_set_begin_offsets ),
        m_particle_set_end_offsets( other.m_particle_set_end_offsets ),
        m_beam_monitor_indices( other.m_beam_monitor_indices ),
        m_my_output_buffer( nullptr ),
        m_my_particles_addr_buffer( nullptr ),
        m_my_elem_by_elem_config( nullptr ),
        m_ptr_particles_buffer( other.m_ptr_particles_buffer ),
        m_ptr_beam_elem_buffer( other.m_ptr_beam_elem_buffer ),
        m_ptr_output_buffer( nullptr ),
        m_ptr_c_particles_buffer( other.m_ptr_c_particles_buffer ),
        m_ptr_c_beam_elem_buffer( other.m_ptr_c_beam_elem_buffer ),
        m_ptr_c_output_buffer( nullptr ),
        m_be_mon_output_buffer_offset( other.m_be_mon_output_buffer_offset ),
        m_elem_by_elem_output_offset( other.m_elem_by_elem_output_offset ),
        m_num_particle_sets_in_buffer( other.m_num_particle_sets_in_buffer ),
        m_num_beam_elements_in_buffer( other.m_num_beam_elements_in_buffer ),
        m_default_elem_by_elem_order( other.m_default_elem_by_elem_order ),
        m_total_num_particles( other.m_total_num_particles ),
        m_total_num_particles_in_sets( other.m_total_num_particles_in_sets ),
        m_min_particle_id( other.m_min_particle_id ),
        m_max_particle_id( other.m_max_particle_id ),
        m_min_element_id( other.m_min_element_id ),
        m_max_element_id( other.m_max_element_id ),
        m_min_initial_turn_id( other.m_min_initial_turn_id ),
        m_max_initial_turn_id( other.m_max_initial_turn_id ),
        m_until_turn_elem_by_elem( other.m_until_turn_elem_by_elem ),
        m_collect_flags( other.m_collect_flags ),
        m_clear_prepare_reset_flags( other.m_clear_prepare_reset_flags ),
        m_clear_all_flags( other.m_clear_all_flags ),
        m_default_elem_by_elem_rolling( other.m_default_elem_by_elem_rolling ),
        m_has_beam_monitor_output( other.m_has_beam_monitor_output ),
        m_has_elem_by_elem_output( other.m_has_elem_by_elem_output ),
        m_has_particle_addresses( other.m_has_particle_addresses ),
        m_requires_collect( other.m_requires_collect ),
        m_uses_controller( other.m_uses_controller ),
        m_uses_arguments( other.m_uses_arguments )
    {
        using elem_by_elem_config_t = TrackJobBaseNew::elem_by_elem_config_t;

        if( other.ownsOutputBuffer() )
        {
            this->m_my_output_buffer.reset( new
                _this_t::buffer_t( *other.ptrOutputBuffer() ) );

            this->m_ptr_output_buffer = this->m_my_output_buffer.get();
            this->m_ptr_c_output_buffer =
                ( this->m_ptr_output_buffer != nullptr )
                    ? this->m_ptr_output_buffer->getCApiPtr() : nullptr;
        }
        else if( other.hasOutputBuffer() )
        {
            this->m_ptr_output_buffer = other.m_ptr_output_buffer;
            this->m_ptr_c_output_buffer = other.m_ptr_c_output_buffer;
        }

        if( other.m_my_particles_addr_buffer.get() != nullptr )
        {
            this->m_my_particles_addr_buffer.reset(
                new _this_t::buffer_t(
                    *other.doGetPtrParticlesAddrBuffer() ) );
        }
        else
        {
            this->m_my_particles_addr_buffer.reset(
                new _this_t::buffer_t );
        }

        if( other.m_my_elem_by_elem_config.get() != nullptr )
        {
            this->m_my_elem_by_elem_config.reset(
                new elem_by_elem_config_t );

            *this->m_my_elem_by_elem_config =  *other.ptrElemByElemConfig();
        }
    }

    TrackJobBaseNew::TrackJobBaseNew(
        TrackJobBaseNew&& other ) SIXTRL_NOEXCEPT :
        st::ArchDebugBase( std::move( other ) ),
        m_particle_set_indices( std::move( other.m_particle_set_indices ) ),
        m_particle_set_begin_offsets(
            std::move( other.m_particle_set_begin_offsets ) ),
        m_particle_set_end_offsets(
            std::move( other.m_particle_set_end_offsets ) ),
        m_beam_monitor_indices( std::move( other.m_beam_monitor_indices ) ),
        m_my_output_buffer( std::move( other.m_my_output_buffer ) ),
        m_my_particles_addr_buffer(
            std::move( other.m_my_particles_addr_buffer ) ),
        m_my_elem_by_elem_config( std::move( other.m_my_elem_by_elem_config ) ),
        m_ptr_particles_buffer( std::move( other.m_ptr_particles_buffer ) ),
        m_ptr_beam_elem_buffer( std::move( other.m_ptr_beam_elem_buffer ) ),
        m_ptr_output_buffer( std::move( other.m_ptr_output_buffer ) ),
        m_ptr_c_particles_buffer( std::move( other.m_ptr_c_particles_buffer )),
        m_ptr_c_beam_elem_buffer( std::move( other.m_ptr_c_beam_elem_buffer ) ),
        m_ptr_c_output_buffer( std::move( other.m_ptr_c_output_buffer ) ),
        m_be_mon_output_buffer_offset(
            std::move( other.m_be_mon_output_buffer_offset ) ),
        m_elem_by_elem_output_offset(
            std::move( other.m_elem_by_elem_output_offset ) ),
        m_num_particle_sets_in_buffer(
            std::move( other.m_num_particle_sets_in_buffer ) ),
        m_num_beam_elements_in_buffer(
            std::move( other.m_num_beam_elements_in_buffer ) ),
        m_default_elem_by_elem_order(
            std::move( other.m_default_elem_by_elem_order ) ),
        m_total_num_particles( std::move( other.m_total_num_particles ) ),
        m_total_num_particles_in_sets( std::move(
            other.m_total_num_particles_in_sets ) ),
        m_min_particle_id( std::move( other.m_min_particle_id ) ),
        m_max_particle_id( std::move( other.m_max_particle_id ) ),
        m_min_element_id( std::move( other.m_min_element_id ) ),
        m_max_element_id( std::move( other.m_max_element_id ) ),
        m_min_initial_turn_id( std::move( other.m_min_initial_turn_id ) ),
        m_max_initial_turn_id( std::move( other.m_max_initial_turn_id ) ),
        m_until_turn_elem_by_elem( std::move(
            other.m_until_turn_elem_by_elem ) ),
        m_collect_flags( std::move( other.m_collect_flags ) ),
        m_clear_prepare_reset_flags(
            std::move( other.m_clear_prepare_reset_flags ) ),
        m_clear_all_flags( std::move( other.m_clear_all_flags ) ),
        m_default_elem_by_elem_rolling(
            std::move( other.m_default_elem_by_elem_rolling ) ),
        m_has_beam_monitor_output(
            std::move( other.m_has_beam_monitor_output ) ),
        m_has_elem_by_elem_output(
            std::move( other.m_has_elem_by_elem_output ) ),
        m_has_particle_addresses(
            std::move( other.m_has_particle_addresses ) ),
        m_requires_collect( std::move( other.m_requires_collect ) ),
        m_uses_controller( std::move( other.m_uses_controller ) ),
        m_uses_arguments( std::move( other.m_uses_arguments ) )
    {
        other.doClearParticlesStructuresBaseImpl();
        other.doClearBeamElementsStructuresBaseImpl();
        other.doClearOutputStructuresBaseImpl();
    }

    TrackJobBaseNew& TrackJobBaseNew::operator=( TrackJobBaseNew const& rhs )
    {
        if( ( this != &rhs ) && ( this->isArchCompatibleWith( rhs ) ) )
        {
            st::ArchDebugBase::operator=( rhs );

            this->m_particle_set_indices = rhs.m_particle_set_indices;

            this->m_particle_set_begin_offsets =
                rhs.m_particle_set_begin_offsets;

            this->m_particle_set_end_offsets =
                rhs.m_particle_set_end_offsets;

            this->m_beam_monitor_indices = rhs.m_beam_monitor_indices;

            /* TODO: Implement (re-)assigning of output buffers for beam-
             *       monitors after copying the outputbuffer! */

            if( rhs.ownsOutputBuffer() )
            {
                if( this->ownsOutputBuffer() )
                {
                    SIXTRL_ASSERT( this->m_my_output_buffer.get() != nullptr );
                    this->m_my_output_buffer->deepCopyFrom(
                        rhs.ptrCOutputBuffer() );

                    this->m_ptr_output_buffer = this->m_my_output_buffer.get();

                    if( this->m_ptr_output_buffer != nullptr )
                    {
                        this->m_ptr_c_output_buffer =
                            this->m_ptr_output_buffer->getCApiPtr();
                    }
                }
                else
                {
                    this->m_ptr_output_buffer = rhs.m_ptr_output_buffer;
                    this->m_ptr_c_output_buffer = rhs.m_ptr_c_output_buffer;
                }
            }
            else if( rhs.hasOutputBuffer() )
            {
                if( this->ownsOutputBuffer() )
                {
                    this->m_my_output_buffer.reset( nullptr );
                }

                this->m_ptr_output_buffer = rhs.m_ptr_output_buffer;
                this->m_ptr_c_output_buffer = rhs.m_ptr_c_output_buffer;
            }

            if( rhs.m_my_particles_addr_buffer.get() != nullptr )
            {
                this->m_my_particles_addr_buffer.reset(
                    new _this_t::buffer_t(
                        *rhs.m_my_particles_addr_buffer ) );
            }
            else
            {
                this->m_my_particles_addr_buffer.reset(
                    new _this_t::buffer_t );
            }

            if( rhs.m_my_elem_by_elem_config.get() != nullptr )
            {
                this->m_my_elem_by_elem_config.reset(
                    new elem_by_elem_config_t );

                *this->m_my_elem_by_elem_config = *rhs.ptrElemByElemConfig();
            }
            else if( this->m_my_elem_by_elem_config.get() != nullptr )
            {
                this->m_my_elem_by_elem_config.reset( nullptr );
            }

            this->m_ptr_particles_buffer = rhs.m_ptr_particles_buffer;
            this->m_ptr_beam_elem_buffer = rhs.m_ptr_beam_elem_buffer;

            this->m_ptr_c_particles_buffer = rhs.m_ptr_c_particles_buffer;
            this->m_ptr_c_beam_elem_buffer = rhs.m_ptr_c_beam_elem_buffer;

            this->m_be_mon_output_buffer_offset =
                rhs.m_be_mon_output_buffer_offset;

            this->m_elem_by_elem_output_offset =
                rhs.m_elem_by_elem_output_offset;

            this->m_num_particle_sets_in_buffer =
                rhs.m_num_particle_sets_in_buffer;

            this->m_num_beam_elements_in_buffer =
                rhs.m_num_beam_elements_in_buffer;

            this->m_default_elem_by_elem_order =
                rhs.m_default_elem_by_elem_order;

            this->m_total_num_particles = rhs.m_total_num_particles;

            this->m_total_num_particles_in_sets =
                rhs.m_total_num_particles_in_sets;

            this->m_min_particle_id = rhs.m_min_particle_id;
            this->m_max_particle_id = rhs.m_max_particle_id;
            this->m_min_element_id  = rhs.m_min_element_id;
            this->m_max_element_id  = rhs.m_max_element_id;
            this->m_min_initial_turn_id = rhs.m_min_initial_turn_id;
            this->m_max_initial_turn_id = rhs.m_max_initial_turn_id;
            this->m_until_turn_elem_by_elem = rhs.m_until_turn_elem_by_elem;
            this->m_collect_flags = rhs.m_collect_flags;

            this->m_clear_prepare_reset_flags =
                rhs.m_clear_prepare_reset_flags;

            this->m_clear_all_flags = rhs.m_clear_all_flags;

            this->m_default_elem_by_elem_rolling =
                rhs.m_default_elem_by_elem_rolling;

            this->m_has_beam_monitor_output = rhs.m_has_beam_monitor_output;
            this->m_has_elem_by_elem_output = rhs.m_has_elem_by_elem_output;
            this->m_has_particle_addresses  = rhs.m_has_particle_addresses;

            this->m_requires_collect = rhs.m_requires_collect;
            this->m_uses_controller = rhs.m_uses_controller;
            this->m_uses_arguments = rhs.m_uses_arguments;
        }

        return *this;
    }

    TrackJobBaseNew& TrackJobBaseNew::operator=(
        TrackJobBaseNew&& rhs ) SIXTRL_NOEXCEPT
    {
        if( this != &rhs )
        {
            st::ArchDebugBase::operator=( std::move( rhs ) );

            this->m_particle_set_indices =
                std::move( rhs.m_particle_set_indices );

            this->m_particle_set_begin_offsets =
                std::move( rhs.m_particle_set_begin_offsets );

            this->m_particle_set_end_offsets =
                std::move( rhs.m_particle_set_end_offsets );

            this->m_beam_monitor_indices =
                std::move( rhs.m_beam_monitor_indices );

            this->m_my_output_buffer = std::move( rhs.m_my_output_buffer );

            this->m_my_particles_addr_buffer =
                std::move( rhs.m_my_particles_addr_buffer );

            this->m_my_elem_by_elem_config =
                std::move( rhs.m_my_elem_by_elem_config );

            this->m_ptr_particles_buffer =
                std::move( rhs.m_ptr_particles_buffer );

            this->m_ptr_beam_elem_buffer =
                std::move( rhs.m_ptr_beam_elem_buffer );

            this->m_ptr_output_buffer = std::move( rhs.m_ptr_output_buffer );

            this->m_ptr_c_particles_buffer =
                std::move( rhs.m_ptr_c_particles_buffer );

            this->m_ptr_c_beam_elem_buffer =
                std::move( rhs.m_ptr_c_beam_elem_buffer );

            this->m_be_mon_output_buffer_offset = std::move(
                rhs.m_be_mon_output_buffer_offset );

            this->m_elem_by_elem_output_offset = std::move(
                rhs.m_elem_by_elem_output_offset );

            this->m_num_particle_sets_in_buffer =
                std::move( rhs.m_num_particle_sets_in_buffer );

            this->m_num_beam_elements_in_buffer =
                std::move( rhs.m_num_beam_elements_in_buffer );

            this->m_default_elem_by_elem_order = std::move(
                rhs.m_default_elem_by_elem_order );

            this->m_total_num_particles =
                std::move( rhs.m_total_num_particles );

            this->m_total_num_particles_in_sets =
                std::move( rhs.m_total_num_particles_in_sets );

            this->m_min_particle_id = std::move( rhs.m_min_particle_id );
            this->m_max_particle_id = std::move( rhs.m_max_particle_id );
            this->m_min_element_id  = std::move( rhs.m_min_element_id );
            this->m_max_element_id  = std::move( rhs.m_max_element_id );

            this->m_min_initial_turn_id =
                std::move( rhs.m_min_initial_turn_id );

            this->m_max_initial_turn_id =
                std::move( rhs.m_max_initial_turn_id );

            this->m_until_turn_elem_by_elem =
                std::move( rhs.m_until_turn_elem_by_elem );

            this->m_collect_flags = std::move( rhs.m_collect_flags );

            this->m_clear_prepare_reset_flags =
                std::move( rhs.m_clear_prepare_reset_flags );

            this->m_clear_all_flags =
                std::move( rhs.m_clear_all_flags );

            this->m_default_elem_by_elem_rolling =
                std::move( rhs.m_default_elem_by_elem_rolling );

            this->m_has_beam_monitor_output =
                std::move( rhs.m_has_beam_monitor_output );

            this->m_has_elem_by_elem_output =
                std::move( rhs.m_has_elem_by_elem_output );

            this->m_has_particle_addresses =
                std::move( rhs.m_has_particle_addresses );

            this->m_requires_collect = std::move( rhs.m_requires_collect );
            this->m_uses_controller = std::move( rhs.m_uses_controller );
            this->m_uses_arguments = std::move( rhs.m_uses_arguments );

            rhs.doClearParticlesStructuresBaseImpl();
            rhs.doClearBeamElementsStructuresBaseImpl();
            rhs.doClearOutputStructuresBaseImpl();
        }

        return *this;
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::doClear( _this_t::clear_flag_t const flags )
    {
        if( _this_t::IsClearFlagSet(
                flags, st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES ) )
        {
            this->doClearParticlesStructures();
        }

        if( _this_t::IsClearFlagSet(
                flags, st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES ) )
        {
            this->doClearBeamElementsStructures();
        }

        if( _this_t::IsClearFlagSet(
                flags,st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES ) )
        {
            this->doClearOutputStructures();
        }

        return;
    }

    _this_t::collect_flag_t TrackJobBaseNew::doCollect(
        _this_t::collect_flag_t const flags )
    {
        return ( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    _this_t::push_flag_t TrackJobBaseNew::doPush(
        _this_t::collect_flag_t const flags )
    {
        return ( flags & st::TRACK_JOB_PUSH_ALL );
    }

    /* --------------------------------------------------------------------- */

    _this_t::status_t TrackJobBaseNew::doPrepareParticlesStructures(
        _this_t::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using size_t = _this_t::size_type;
        using p_index_t = _this_t::particle_index_t;

        SIXTRL_STATIC_VAR size_t const ZERO = size_t{ 0 };
        SIXTRL_STATIC_VAR size_t const ONE  = size_t{ 1 };

        size_t const nn = this->numParticleSets();
        size_t const num_psets = ::NS(Buffer_get_num_of_objects)( pb );

        if( ( pb != nullptr ) && ( ( nn > ZERO ) || ( nn == ZERO ) ) &&
            ( !::NS(Buffer_needs_remapping)( pb ) ) && ( num_psets >= nn ) )
        {
            size_t const first_index = ( nn > ZERO )
                ? this->particleSetIndex( ZERO ) : ZERO;

            p_index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
                      min_turn_id, max_turn_id;

            if( ( nn <= ONE ) && ( first_index == ZERO ) )
            {
                status = ::NS(Particles_get_min_max_attributes)(
                        ::NS(Particles_buffer_get_const_particles)( pb, ZERO ),
                        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id,
                        &min_turn_id, &max_turn_id );
            }
            else if( nn > ZERO )
            {
                status = ::NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
                    pb, nn, this->particleSetIndicesBegin(),
                    &min_part_id, &max_part_id, &min_elem_id, &max_elem_id,
                    &min_turn_id, &max_turn_id );
            }

            if( this->doGetPtrParticlesAddrBuffer() == nullptr )
            {
                _this_t::ptr_particles_addr_buffer_t partaddr_buffer_store(
                    new _this_t::buffer_t );

                this->doUpdateStoredParticlesAddrBuffer(
                    std::move( partaddr_buffer_store ) );
            }

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->doGetPtrParticlesAddrBuffer() != nullptr )  )
            {
                status = ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
                    this->doGetPtrParticlesAddrBuffer()->getCApiPtr(), pb );
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                this->doSetMinParticleId( min_part_id );
                this->doSetMaxParticleId( max_part_id );

                this->doSetMinElementId( min_elem_id );
                this->doSetMaxElementId( max_elem_id );

                this->doSetMinInitialTurnId( min_turn_id );
                this->doSetMaxInitialTurnId( max_turn_id );
            }
        }

        return status;
    }

    void TrackJobBaseNew::doClearParticlesStructures()
    {
        this->doClearParticlesStructuresBaseImpl();
    }

    _this_t::status_t TrackJobBaseNew::doPrepareBeamElementsStructures(
        _this_t::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using size_t        = _this_t::size_type;
        using p_index_t     = _this_t::particle_index_t;
        using buf_size_t    = ::NS(buffer_size_t);
        using obj_ptr_t     = SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object)*;
        using ptr_t         = SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*;

        SIXTRL_STATIC_VAR size_t const ZERO = size_t{ 0 };
        SIXTRL_STATIC_VAR uintptr_t const UZERO = uintptr_t{ 0 };

        if( ( belems != nullptr ) &&
            ( !::NS(Buffer_needs_remapping)( belems ) ) &&
            ( ::NS(Buffer_get_num_of_objects)( belems ) > ZERO ) &&
            ( ::NS(BeamElements_is_beam_elements_buffer)( belems ) ) )
        {
            p_index_t const start_be_id = p_index_t{ 0 };
            buf_size_t  num_e_by_e_objs = buf_size_t{ 0 };
            p_index_t min_elem_id = this->minElementId();
            p_index_t max_elem_id = this->maxElementId();

            status = ::NS(ElemByElemConfig_find_min_max_element_id_from_buffer)(
                belems, &min_elem_id, &max_elem_id, &num_e_by_e_objs,
                    start_be_id );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                buf_size_t num_be_monitors = buf_size_t{ 0 };
                std::vector< size_t > be_mon_indices( num_e_by_e_objs, ZERO );

                status = ::NS(BeamMonitor_get_beam_monitor_indices_from_buffer)(
                    belems, be_mon_indices.size(), be_mon_indices.data(),
                        &num_be_monitors );

                SIXTRL_ASSERT( num_be_monitors <= be_mon_indices.size() );

                auto ind_end = be_mon_indices.begin();

                if( num_be_monitors > buf_size_t{ 0 } )
                {
                    std::advance( ind_end, num_be_monitors );
                }

                this->doSetBeamMonitorIndices( be_mon_indices.begin(),ind_end);
                SIXTRL_ASSERT( num_be_monitors == this->numBeamMonitors() );

                this->doSetMinElementId( min_elem_id );
                this->doSetMaxElementId( max_elem_id );
            }

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->numBeamMonitors() > ZERO ) &&
                ( this->ptrCParticlesBuffer() != nullptr ) &&
                ( this->minParticleId() <= this->maxParticleId() ) )
            {
                auto it  = this->beamMonitorIndicesBegin();
                auto end = this->beamMonitorIndicesEnd();

                p_index_t const min_part_id = this->minParticleId();
                p_index_t const max_part_id = this->maxParticleId();

                for( ; it != end ; ++it )
                {
                    obj_ptr_t obj = ::NS(Buffer_get_object)( belems, *it );
                    uintptr_t const addr = static_cast< uintptr_t >(
                        ::NS(Object_get_begin_addr)( obj ) );

                    if( ( obj != nullptr ) && ( addr != UZERO ) &&
                        ( ::NS(Object_get_type_id)( obj ) ==
                          ::NS(OBJECT_TYPE_BEAM_MONITOR) ) )
                    {
                        ptr_t m = reinterpret_cast< ptr_t >( addr );
                        ::NS(BeamMonitor_set_min_particle_id)( m,min_part_id );
                        ::NS(BeamMonitor_set_max_particle_id)( m,max_part_id );
                    }
                    else
                    {
                        status = st::ARCH_STATUS_GENERAL_FAILURE;
                        break;
                    }
                }

                this->m_num_beam_elements_in_buffer = num_e_by_e_objs;
            }
        }

        return status;
    }

    void TrackJobBaseNew::doClearBeamElementsStructures()
    {
        this->doClearBeamElementsStructuresBaseImpl();
    }

    _this_t::status_t TrackJobBaseNew::doPrepareOutputStructures(
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        _this_t::size_type const until_turn_elem_by_elem )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using size_t                 = _this_t::size_type;
        using buffer_t               = _this_t::buffer_t;
        using c_buffer_t             = _this_t::c_buffer_t;
        using buf_size_t             = ::NS(buffer_size_t);
        using elem_by_elem_config_t  = TrackJobBaseNew::elem_by_elem_config_t;
        using ptr_output_buffer_t    = TrackJobBaseNew::ptr_output_buffer_t;
        using par_index_t            = _this_t::particle_index_t;

        using ptr_elem_by_elem_config_t =
            TrackJobBaseNew::ptr_elem_by_elem_config_t;

        SIXTRL_ASSERT( particles_buffer != nullptr );
        SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( particles_buffer ) );

        SIXTRL_ASSERT( ( ::NS(Buffer_get_num_of_objects)( particles_buffer ) ==
            size_t{ 0 } ) || ( ::NS(Buffer_is_particles_buffer)(
                particles_buffer ) ) );

        SIXTRL_ASSERT( beam_elements_buffer != nullptr );
        SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( beam_elements_buffer ) );

        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( beam_elements_buffer )
            > size_t{ 0 } );

        SIXTRL_ASSERT( ::NS(BeamElements_is_beam_elements_buffer)(
            beam_elements_buffer ) );

        c_buffer_t* output_buffer = ptr_output_buffer;

        if( output_buffer == nullptr )
        {
            if( !this->hasOutputBuffer() )
            {
                SIXTRL_ASSERT( !this->ownsOutputBuffer() );
                ptr_output_buffer_t managed_output_buffer( new buffer_t );

                SIXTRL_ASSERT(  managed_output_buffer.get() != nullptr );
                output_buffer = managed_output_buffer.get()->getCApiPtr();
                SIXTRL_ASSERT( output_buffer != nullptr );

                this->doUpdateStoredOutputBuffer(
                    std::move( managed_output_buffer ) );

                SIXTRL_ASSERT( managed_output_buffer.get() == nullptr );
                SIXTRL_ASSERT( this->ownsOutputBuffer() );
                SIXTRL_ASSERT( this->ptrCOutputBuffer() == output_buffer );
            }
            else
            {
                output_buffer = this->ptrCOutputBuffer();
            }

            SIXTRL_ASSERT( this->hasOutputBuffer() );
        }
        else
        {
            if( !this->hasOutputBuffer() )
            {
                this->doSetPtrCOutputBuffer( output_buffer );
            }
            else if( !this->ownsOutputBuffer() )
            {
                this->doSetPtrCOutputBuffer( output_buffer );
            }
            else if( ( this->ownsOutputBuffer() ) &&
                     ( this->ptrCOutputBuffer() != nullptr ) &&
                     ( this->ptrCOutputBuffer() != output_buffer ) )
            {
                ptr_output_buffer_t dummy( nullptr );
                this->doUpdateStoredOutputBuffer( std::move( dummy ) );
                this->doSetPtrCOutputBuffer( output_buffer );
            }
            else
            {
                return status;
            }
        }

        if( output_buffer != nullptr )
        {
            SIXTRL_STATIC_VAR const size_t ZERO = size_t{ 0 };
            SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( output_buffer ) );

            SIXTRL_ASSERT(
                ( ::NS(Buffer_get_num_of_objects)( output_buffer ) == ZERO )||
                ( ::NS(Buffer_is_particles_buffer)( output_buffer ) ) );

            buf_size_t  elem_by_elem_out_idx_offset = buf_size_t{ 0 };
            buf_size_t  be_monitor_out_idx_offset   = buf_size_t{ 0 };
            par_index_t max_elem_by_elem_turn_id    = par_index_t{ 0 };

            status = ::NS(OutputBuffer_prepare_detailed)(
                beam_elements_buffer, output_buffer,
                this->minParticleId(), this->maxParticleId(),
                this->minElementId(),  this->maxElementId(),
                this->minInitialTurnId(), this->maxInitialTurnId(),
                until_turn_elem_by_elem,
                &elem_by_elem_out_idx_offset, &be_monitor_out_idx_offset,
                &max_elem_by_elem_turn_id );

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( until_turn_elem_by_elem > ZERO ) &&
                ( this->minInitialTurnId() >= par_index_t{ 0 } ) &&
                ( max_elem_by_elem_turn_id >= this->minInitialTurnId() ) &&
                ( until_turn_elem_by_elem > static_cast< buf_size_t >(
                    this->minInitialTurnId() ) ) )
            {
                ptr_elem_by_elem_config_t conf( new elem_by_elem_config_t );
                ::NS(ElemByElemConfig_preset)( conf.get() );

                status = ::NS(ElemByElemConfig_init_detailed)(
                    conf.get(), this->defaultElemByElemOrder(),
                    this->minParticleId(), this->maxParticleId(),
                    this->minElementId(),  this->maxElementId(),
                    this->minInitialTurnId(), max_elem_by_elem_turn_id,
                    this->defaultElemByElemRolling() );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    this->doUpdateStoredElemByElemConfig( std::move( conf ) );
                    this->doSetUntilTurnElemByElem(
                        until_turn_elem_by_elem );

                    SIXTRL_ASSERT( this->hasElemByElemConfig() );
                }
            }
            else if( status == st::ARCH_STATUS_SUCCESS )
            {
                ptr_elem_by_elem_config_t dummy( nullptr );
                this->doUpdateStoredElemByElemConfig( std::move( dummy ) );
                this->doSetUntilTurnElemByElem( ZERO );

                status = ( !this->hasElemByElemConfig() )
                    ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                this->doSetBeamMonitorOutputBufferOffset(
                    be_monitor_out_idx_offset );

                this->doSetElemByElemOutputIndexOffset(
                    elem_by_elem_out_idx_offset );
            }
        }

        return status;
    }

    void TrackJobBaseNew::doClearOutputStructures()
    {
        this->doClearOutputStructuresBaseImpl();
    }

    _this_t::status_t TrackJobBaseNew::doAssignOutputBufferToBeamMonitors(
        _this_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        _this_t::particle_index_t const min_turn_id,
        _this_t::size_type const output_buffer_offset_index )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        this->doSetBeamMonitorOutputEnabledFlag( false );

        if( ( output_buffer != nullptr ) && ( beam_elem_buffer != nullptr ) &&
            ( this->numBeamMonitors() > _this_t::size_type{ 0 } ) &&
            ( min_turn_id >= _this_t::particle_index_t{ 0 } ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
              output_buffer_offset_index ) )
        {
            if( !this->isInDebugMode() )
            {
                status = ::NS(BeamMonitor_assign_output_buffer_from_offset)(
                    beam_elem_buffer, output_buffer, min_turn_id,
                    output_buffer_offset_index );
            }
            else if( this->doGetPtrLocalDebugRegister() != nullptr )
            {
                *( this->doGetPtrLocalDebugRegister() ) =
                    st::ARCH_DEBUGGING_GENERAL_FAILURE;

                SIXTRL_ASSERT( ::NS(Buffer_get_slot_size)( output_buffer ) ==
                    ::NS(Buffer_get_slot_size)( beam_elem_buffer ) );

                status = NS(BeamMonitor_assign_output_buffer_from_offset_debug)(
                    beam_elem_buffer, output_buffer, min_turn_id,
                        output_buffer_offset_index,
                            this->doGetPtrLocalDebugRegister() );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    status = ::NS(DebugReg_get_stored_arch_status)(
                        *this->doGetPtrLocalDebugRegister() );
                }
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                this->doSetBeamMonitorOutputEnabledFlag( true );
            }
        }

        return status;
    }


    _this_t::status_t TrackJobBaseNew::doAssignOutputBufferToElemByElemConfig(
        _this_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
        _this_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        _this_t::size_type const output_buffer_offset_index )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( elem_by_elem_config != nullptr ) &&
            ( output_buffer != nullptr ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
                output_buffer_offset_index ) )
        {
            this->doSetElemByElemOutputEnabledFlag( false );

            if( this->isInDebugMode() )
            {
                status = ::NS(ElemByElemConfig_assign_output_buffer)(
                    elem_by_elem_config, output_buffer,
                        output_buffer_offset_index );
            }
            else if( this->doGetPtrLocalDebugRegister() != nullptr )
            {
                *( this->doGetPtrLocalDebugRegister() ) =
                    st::ARCH_DEBUGGING_GENERAL_FAILURE;

                status = ::NS(ElemByElemConfig_assign_output_buffer_debug)(
                    elem_by_elem_config, output_buffer,
                        output_buffer_offset_index,
                            this->doGetPtrLocalDebugRegister() );

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    status = ::NS(DebugReg_get_stored_arch_status)(
                        *this->doGetPtrLocalDebugRegister() );
                }
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                this->doSetElemByElemOutputEnabledFlag( true );
            }
        }

        return status;
    }

    _this_t::clear_flag_t TrackJobBaseNew::doPrepareResetClearFlags(
        const _this_t::c_buffer_t *const SIXTRL_RESTRICT pbuffer,
        _this_t::size_type const num_psets,
        _this_t::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        const _this_t::c_buffer_t *const SIXTRL_RESTRICT belems_buffer,
        const _this_t::c_buffer_t *const SIXTRL_RESTRICT output_buffer,
        _this_t::size_type const )
    {
        using clear_flag_t = _this_t::clear_flag_t;
        clear_flag_t clear_flags = this->doGetDefaultPrepareResetClearFlags();

        bool const has_same_particle_sets = (
            ( pset_indices_begin != nullptr ) &&
            ( this->numParticleSets() == num_psets ) &&
            ( ( pset_indices_begin == this->particleSetIndicesBegin() ) ||
              ( std::equal( pset_indices_begin, pset_indices_begin + num_psets,
                            this->particleSetIndicesBegin() ) ) ) );

        if( ( _this_t::IsClearFlagSet( clear_flags,
                st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES ) ) &&
            ( pbuffer != nullptr ) &&
            ( pbuffer == this->ptrCParticlesBuffer() ) &&
            ( has_same_particle_sets ) )
        {
            clear_flags = _this_t::UnsetClearFlag( clear_flags,
               st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES );
        }

        if( ( _this_t::IsClearFlagSet( clear_flags,
                st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES ) ) &&
            ( belems_buffer != nullptr ) &&
            ( belems_buffer == this->ptrCBeamElementsBuffer() ) )
        {
            clear_flags = _this_t::UnsetClearFlag( clear_flags,
               st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES );
        }

        if( ( _this_t::IsClearFlagSet( clear_flags,
                st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES ) ) &&
            ( output_buffer != nullptr ) &&
            ( output_buffer == this->ptrCOutputBuffer() ) )
        {
            clear_flags = _this_t::UnsetClearFlag( clear_flags,
               st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES );
        }

        return clear_flags;
    }

    _this_t::status_t TrackJobBaseNew::doReset(
        _this_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        _this_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        _this_t::size_type const until_turn_elem_by_elem )
    {
        using output_buffer_flag_t = _this_t::output_buffer_flag_t;

        _this_t::status_t status =
            this->doPrepareParticlesStructures( particles_buffer );

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            status = this->doPrepareBeamElementsStructures( beam_elem_buffer );
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            this->doSetPtrCParticlesBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elem_buffer );

            output_buffer_flag_t const out_buffer_flags =
            ::NS(OutputBuffer_required_for_tracking_of_particle_sets)(
                particles_buffer, this->numParticleSets(),
                    this->particleSetIndicesBegin(), beam_elem_buffer,
                        until_turn_elem_by_elem );

            bool const requires_output_buffer =
                ::NS(OutputBuffer_requires_output_buffer)( out_buffer_flags );

            if( requires_output_buffer )
            {
                status = this->doPrepareOutputStructures( particles_buffer,
                    beam_elem_buffer, output_buffer, until_turn_elem_by_elem );
            }

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( this->hasOutputBuffer() ) && ( requires_output_buffer ) )
            {
                if( ::NS(OutputBuffer_requires_elem_by_elem_output)(
                        out_buffer_flags ) )
                {
                    status = this->doAssignOutputBufferToElemByElemConfig(
                        this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                            this->elemByElemOutputBufferOffset() );
                }

                if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                    ( ::NS(OutputBuffer_requires_beam_monitor_output)(
                        out_buffer_flags ) ) )
                {
                    status = this->doAssignOutputBufferToBeamMonitors(
                        beam_elem_buffer, this->ptrCOutputBuffer(),
                        this->minInitialTurnId(),
                        this->beamMonitorsOutputBufferOffset() );
                }
            }

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( output_buffer != nullptr ) && ( !this->hasOutputBuffer() ) )
            {
                /* ( !this->ownsOutputBuffer() ) */
                this->doSetPtrCOutputBuffer( output_buffer );
            }
        }

        return status;
    }

    _this_t::status_t TrackJobBaseNew::doAssignNewOutputBuffer(
        c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        _this_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        using flags_t = _this_t::output_buffer_flag_t;

        flags_t const flags =
            ::NS(OutputBuffer_required_for_tracking_of_particle_sets)(
            this->ptrCParticlesBuffer(), this->numParticleSets(),
                this->particleSetIndicesBegin(),
                this->ptrCBeamElementsBuffer(), this->numElemByElemTurns() );

        bool const requires_output_buffer =
            ::NS(OutputBuffer_requires_output_buffer)( flags );

        if( requires_output_buffer )
        {
            status = this->doPrepareOutputStructures(
                this->ptrCParticlesBuffer(), this->ptrCBeamElementsBuffer(),
                ptr_output_buffer, this->numElemByElemTurns() );
        }

        if( ( status == st::ARCH_STATUS_SUCCESS ) &&
            ( requires_output_buffer ) && ( this->hasOutputBuffer() ) )
        {
            if( ( ::NS(OutputBuffer_requires_beam_monitor_output)( flags ) ) &&
                ( this->hasBeamMonitorOutput() ) )
            {
                status = this->doAssignOutputBufferToBeamMonitors(
                    this->ptrCBeamElementsBuffer(), this->ptrCOutputBuffer(),
                        this->minInitialTurnId(),
                            this->beamMonitorsOutputBufferOffset() );
            }

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( ::NS(OutputBuffer_requires_elem_by_elem_output)( flags ) ) &&
                ( this->hasElemByElemOutput() ) )
            {
                status = this->doAssignOutputBufferToElemByElemConfig(
                    this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                        this->elemByElemOutputBufferOffset() );
            }
        }

        return status;
    }

    /* --------------------------------------------------------------------- */

    _this_t::status_t TrackJobBaseNew::doFetchParticleAddresses()
    {
        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    _this_t::status_t TrackJobBaseNew::doClearParticleAddresses(
        _this_t::size_type const index )
    {
        using status_t = _this_t::status_t;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( this->doGetPtrParticlesAddrBuffer() != nullptr ) &&
            ( this->doGetPtrParticlesAddrBuffer()->getNumObjects() > index ) )
        {
            status = ::NS(ParticlesAddr_buffer_clear_single)(
                this->doGetPtrParticlesAddrBuffer()->getCApiPtr(), index );

            this->doSetHasParticleAddressesFlag( false );
        }

        return status;
    }

    _this_t::status_t TrackJobBaseNew::doClearAllParticleAddresses()
    {
        using status_t = _this_t::status_t;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->doGetPtrParticlesAddrBuffer() != nullptr )
        {
            status = ::NS(ParticlesAddr_buffer_clear_all)(
                this->doGetPtrParticlesAddrBuffer()->getCApiPtr() );

            this->doSetHasParticleAddressesFlag( false );
        }

        return status;
    }

    _this_t::track_status_t TrackJobBaseNew::doTrackUntilTurn(
        _this_t::size_type const )
    {
        return st::TRACK_STATUS_GENERAL_FAILURE;
    }

    TrackJobBaseNew::track_status_t TrackJobBaseNew::doTrackElemByElem(
        _this_t::size_type const until_turn_elem_by_elem )
    {
        return st::TRACK_STATUS_GENERAL_FAILURE;
    }

    TrackJobBaseNew::track_status_t TrackJobBaseNew::doTrackLine(
        _this_t::size_type const, _this_t::size_type const,
            bool const )
    {
        return st::TRACK_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::doSetPtrParticlesBuffer(
        _this_t::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ptr_buffer != nullptr )
        {
            this->m_ptr_c_particles_buffer = ptr_buffer->getCApiPtr();
        }
        else if( ( this->m_ptr_particles_buffer != nullptr ) &&
                 ( this->m_ptr_c_particles_buffer ==
                   this->m_ptr_particles_buffer->getCApiPtr() ) )
        {
            this->m_ptr_c_particles_buffer = nullptr;
        }

        this->m_ptr_particles_buffer = ptr_buffer;
    }

    void TrackJobBaseNew::doSetPtrBeamElementsBuffer(
        _this_t::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ptr_buffer != nullptr )
        {
            this->m_ptr_c_beam_elem_buffer = ptr_buffer->getCApiPtr();
        }
        else if( ( this->m_ptr_beam_elem_buffer != nullptr ) &&
                 ( this->m_ptr_c_beam_elem_buffer ==
                   this->m_ptr_beam_elem_buffer->getCApiPtr() ) )
        {
            this->m_ptr_c_beam_elem_buffer = nullptr;
        }

        this->m_ptr_beam_elem_buffer = ptr_buffer;
    }

    void TrackJobBaseNew::doSetPtrOutputBuffer(
        _this_t::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ptr_buffer != nullptr )
        {
            this->m_ptr_c_output_buffer = ptr_buffer->getCApiPtr();
        }
        else if( ( this->m_ptr_output_buffer != nullptr ) &&
                 ( this->m_ptr_c_output_buffer ==
                   this->m_ptr_output_buffer->getCApiPtr() ) )
        {
            this->m_ptr_c_output_buffer = nullptr;
        }

        this->m_ptr_output_buffer = ptr_buffer;
    }

    void TrackJobBaseNew::doSetCxxBufferPointers(
        _this_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        _this_t::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        _this_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer ) SIXTRL_NOEXCEPT
    {
        this->doSetPtrParticlesBuffer( &particles_buffer );
        this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

        if( ( ptr_output_buffer != nullptr ) && ( this->hasOutputBuffer() ) &&
            ( !this->ownsOutputBuffer() ) &&
            ( ptr_output_buffer->getCApiPtr() == this->ptrCOutputBuffer() ) )
        {
            this->doSetPtrOutputBuffer( ptr_output_buffer );
        }

        return;
    }

    void TrackJobBaseNew::doSetPtrCParticlesBuffer( _this_t::c_buffer_t*
        SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ( this->m_ptr_particles_buffer   != nullptr ) &&
            ( this->m_ptr_c_particles_buffer ==
              this->m_ptr_particles_buffer->getCApiPtr() ) &&
            ( this->m_ptr_particles_buffer->getCApiPtr() != ptr_buffer ) )
        {
            this->m_ptr_particles_buffer = nullptr;
        }

        this->m_ptr_c_particles_buffer = ptr_buffer;
    }

    void TrackJobBaseNew::doSetPtrCBeamElementsBuffer(
        _this_t::c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ( this->m_ptr_beam_elem_buffer != nullptr ) &&
            ( this->m_ptr_c_beam_elem_buffer ==
              this->m_ptr_beam_elem_buffer->getCApiPtr() ) &&
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() != ptr_buffer ) )
        {
            this->m_ptr_beam_elem_buffer = nullptr;
        }

        this->m_ptr_c_beam_elem_buffer = ptr_buffer;
    }

    void TrackJobBaseNew::doSetPtrCOutputBuffer( _this_t::c_buffer_t*
        SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ( this->m_ptr_output_buffer != nullptr ) &&
            ( this->m_ptr_c_output_buffer ==
              this->m_ptr_output_buffer->getCApiPtr() ) &&
            ( this->m_ptr_output_buffer->getCApiPtr() != ptr_buffer ) )
        {
            this->m_ptr_output_buffer = nullptr;
        }

        this->m_ptr_c_output_buffer = ptr_buffer;
    }

    void TrackJobBaseNew::doSetBeamMonitorOutputBufferOffset(
        _this_t::size_type const
            output_buffer_offset ) SIXTRL_NOEXCEPT
    {
        this->m_be_mon_output_buffer_offset = output_buffer_offset;
    }

    void TrackJobBaseNew::doSetElemByElemOutputIndexOffset(
        _this_t::size_type const
            elem_by_elem_output_offset ) SIXTRL_NOEXCEPT
    {
        this->m_elem_by_elem_output_offset = elem_by_elem_output_offset;
    }

    void TrackJobBaseNew::doSetUntilTurnElemByElem(
        _this_t::particle_index_t
            const until_turn_elem_by_elem ) SIXTRL_NOEXCEPT
    {
        this->m_until_turn_elem_by_elem = until_turn_elem_by_elem;
    }

    void TrackJobBaseNew::doSetRequiresCollectFlag(
        bool const requires_collect_flag ) SIXTRL_NOEXCEPT
    {
        this->m_requires_collect = requires_collect_flag;
    }

    void TrackJobBaseNew::doSetBeamMonitorOutputEnabledFlag(
        bool const has_beam_monitor_output ) SIXTRL_NOEXCEPT
    {
        this->m_has_beam_monitor_output = has_beam_monitor_output;
    }

    void TrackJobBaseNew::doSetElemByElemOutputEnabledFlag(
        bool const has_elem_by_elem_output ) SIXTRL_NOEXCEPT
    {
        this->m_has_elem_by_elem_output = has_elem_by_elem_output;
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::doInitDefaultParticleSetIndices()
    {
        this->m_particle_set_indices.clear();
        this->m_particle_set_indices.push_back( _this_t::size_type{ 0 } );

        this->m_particle_set_begin_offsets.clear();
        this->m_particle_set_begin_offsets.push_back(
            _this_t::num_particles_t{ 0 } );

        this->m_particle_set_end_offsets.clear();
        this->m_particle_set_end_offsets.push_back(
            _this_t::num_particles_t{ 0 } );
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::doInitDefaultBeamMonitorIndices()
    {
        this->m_beam_monitor_indices.clear();
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::doSetNumParticleSetsInBuffer(
        _this_t::size_type const num_psets ) SIXTRL_NOEXCEPT
    {
        this->m_num_particle_sets_in_buffer = num_psets;
    }

    SIXTRL_HOST_FN void TrackJobBaseNew::doSetNumBeamElementsInBuffer(
        _this_t::size_type const num_belems ) SIXTRL_NOEXCEPT
    {
        this->m_num_beam_elements_in_buffer = num_belems;
    }

    void TrackJobBaseNew::doSetTotalNumParticles(
        _this_t::num_particles_t const num_particles ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles = num_particles;
    }

    void TrackJobBaseNew::doSetTotalNumParticlesInSets(
        _this_t::num_particles_t const pnum_in_sets ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles_in_sets = pnum_in_sets;
    }

    void TrackJobBaseNew::doSetMinParticleId(
        _this_t::particle_index_t const min_part_id ) SIXTRL_NOEXCEPT
    {
         this->m_min_particle_id = min_part_id;
    }

    void TrackJobBaseNew::doSetMaxParticleId(
        _this_t::particle_index_t const max_part_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_particle_id = max_part_id;
    }

    void TrackJobBaseNew::doSetMinElementId(
        _this_t::particle_index_t const min_elem_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_element_id = min_elem_id;
    }

    void TrackJobBaseNew::doSetMaxElementId(
        _this_t::particle_index_t const max_elem_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_element_id = max_elem_id;
    }

    void TrackJobBaseNew::doSetMinInitialTurnId(
        _this_t::particle_index_t const
            min_initial_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_initial_turn_id = min_initial_turn_id;
    }

    void TrackJobBaseNew::doSetMaxInitialTurnId( particle_index_t const
        max_initial_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_initial_turn_id = max_initial_turn_id;
    }

    /* --------------------------------------------------------------------- */

    _this_t::buffer_t const*
        TrackJobBaseNew::doGetPtrParticlesAddrBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_my_particles_addr_buffer.get();
    }

    _this_t::buffer_t*
    TrackJobBaseNew::doGetPtrParticlesAddrBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_my_particles_addr_buffer.get();
    }

    void TrackJobBaseNew::doUpdateStoredParticlesAddrBuffer(
        TrackJobBaseNew::ptr_particles_addr_buffer_t&&
            ptr_buffer ) SIXTRL_NOEXCEPT
    {
        this->m_my_particles_addr_buffer = std::move( ptr_buffer );
    }

    void TrackJobBaseNew::doSetHasParticleAddressesFlag(
            bool const has_particle_addresses ) SIXTRL_NOEXCEPT
    {
        this->m_has_particle_addresses = has_particle_addresses;
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::doUpdateStoredOutputBuffer(
        TrackJobBaseNew::ptr_output_buffer_t&& ptr_out_buffer ) SIXTRL_NOEXCEPT
    {
        this->doSetPtrOutputBuffer( ptr_out_buffer.get() );
        this->m_my_output_buffer = std::move( ptr_out_buffer );
    }

    void TrackJobBaseNew::doUpdateStoredElemByElemConfig(
        TrackJobBaseNew::ptr_elem_by_elem_config_t&& ptr_conf ) SIXTRL_NOEXCEPT
    {
        this->m_my_elem_by_elem_config = std::move( ptr_conf );
    }

    void TrackJobBaseNew::doSetUsesControllerFlag(
        bool const uses_controller_flag ) SIXTRL_NOEXCEPT
    {
        this->m_uses_controller = uses_controller_flag;
    }

    void TrackJobBaseNew::doSetUsesArgumentsFlag(
        bool const arguments_flag ) SIXTRL_NOEXCEPT
    {
        this->m_uses_arguments = arguments_flag;
    }

    /* --------------------------------------------------------------------- */

    void TrackJobBaseNew::doClearParticlesStructuresBaseImpl() SIXTRL_NOEXCEPT
    {
        using num_particles_t = _this_t::num_particles_t;
        using size_t = _this_t::size_type;

        this->doSetPtrParticlesBuffer( nullptr );
        this->doSetPtrCParticlesBuffer( nullptr );

        if( this->doGetPtrParticlesAddrBuffer() != nullptr )
        {
            this->doGetPtrParticlesAddrBuffer()->clear( true );
            this->doGetPtrParticlesAddrBuffer()->reset();
        }

        this->doSetTotalNumParticles( num_particles_t{ 0 } );
        this->doSetTotalNumParticlesInSets( num_particles_t{ 0 } );
        this->doSetNumParticleSetsInBuffer( size_t{ 0 } );

        this->doInitDefaultParticleSetIndices();

        ::NS(Particles_init_min_max_attributes_for_find)(
            &this->m_min_particle_id, &this->m_max_particle_id,
            &this->m_min_element_id,  &this->m_max_element_id,
            &this->m_min_initial_turn_id, &this->m_max_initial_turn_id );

        this->doSetHasParticleAddressesFlag( false );

        return;
    }

    void TrackJobBaseNew::doClearBeamElementsStructuresBaseImpl() SIXTRL_NOEXCEPT
    {
        using size_t = _this_t::size_type;

        this->doSetPtrBeamElementsBuffer( nullptr );
        this->doSetPtrCBeamElementsBuffer( nullptr );

        this->doInitDefaultBeamMonitorIndices();
        this->doSetNumBeamElementsInBuffer( size_t{ 0 } );
    }

    void TrackJobBaseNew::doClearOutputStructuresBaseImpl() SIXTRL_NOEXCEPT
    {
        using size_t = _this_t::size_type;

        this->doSetPtrOutputBuffer( nullptr );
        this->doSetPtrCOutputBuffer( nullptr );

        this->doSetBeamMonitorOutputEnabledFlag( false );
        this->doSetElemByElemOutputEnabledFlag( false );

        this->doSetBeamMonitorOutputBufferOffset( size_t{ 0 } );
        this->doSetElemByElemOutputIndexOffset( size_t{ 0 } );
        this->doSetUntilTurnElemByElem( size_t{ 0 } );

        this->m_my_output_buffer.reset( nullptr );
        this->m_my_elem_by_elem_config.reset( nullptr );

        this->m_default_elem_by_elem_order   =
            ::NS(ELEM_BY_ELEM_ORDER_DEFAULT);

        this->m_default_elem_by_elem_rolling = true;

        return;
    }

    /* ********************************************************************* */

    TrackJobBaseNew* TrackJobNew_create(
        TrackJobBaseNew::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT conf_str )
    {
        TrackJobBaseNew* ptr_track_job = nullptr;

        switch( arch_id )
        {
            case st::ARCHITECTURE_CUDA:
            {
                #if defined( SIXTRACKLIB_ENABLE_MODULE_CUDA ) && \
                    SIXTRACKLIB_ENABLE_MODULE_CUDA == 1
                ptr_track_job = new st::CudaTrackJob( conf_str );
                #endif /* CUDA */

                break;
            }

            default:
            {
                ptr_track_job = nullptr;
            }
        };

        return ptr_track_job;
    }

    TrackJobBaseNew* TrackJobNew_create(
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str )
    {
        return st::TrackJobNew_create(
            st::ArchInfo_arch_string_to_arch_id( arch_str ), config_str );
    }

    TrackJobBaseNew* TrackJobNew_new(
        char const* SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        char const* SIXTRL_RESTRICT config_str )
    {
        return st::TrackJobNew_new(
            st::ArchInfo_arch_string_to_arch_id( arch_str ), particles_buffer,
                belements_buffer, config_str );
    }

    TrackJobBaseNew* TrackJobNew_new(
        ::NS(arch_id_t) const arch_id,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        char const* SIXTRL_RESTRICT config_str )
    {
        TrackJobBaseNew* ptr_track_job =
            st::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            if( !ptr_track_job->reset( particles_buffer, belements_buffer ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }

    TrackJobBaseNew* TrackJobNew_new(
        char const* SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str )
    {
        return st::TrackJobNew_new(
            st::ArchInfo_arch_string_to_arch_id( arch_str ), particles_buffer,
                belements_buffer, output_buffer, until_turn_elem_by_elem,
                    config_str );
    }

    TrackJobBaseNew* TrackJobNew_new(
        ::NS(arch_id_t) const arch_id,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str )
    {
        TrackJobBaseNew* ptr_track_job =
            st::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            if( !ptr_track_job->reset( particles_buffer, belements_buffer,
                    output_buffer, until_turn_elem_by_elem ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }

    TrackJobBaseNew* TrackJobNew_new(
        char const* SIXTRL_RESTRICT arch_str,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_psets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        char const* SIXTRL_RESTRICT config_str )
    {
        return st::TrackJobNew_new(
            st::ArchInfo_arch_string_to_arch_id( arch_str ), particles_buffer,
                num_psets, pset_indices_begin, belements_buffer,
                    output_buffer, until_turn_elem_by_elem, config_str );
    }

    TrackJobBaseNew* TrackJobNew_new(
        ::NS(arch_id_t) const SIXTRL_RESTRICT arch_id,
        ::NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
        ::NS(buffer_size_t) const num_psets,
        ::NS(buffer_size_t) const* SIXTRL_RESTRICT pset_indices_begin,
        ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(Buffer)* SIXTRL_RESTRICT output_buffer,
        ::NS(buffer_size_t) const until_turn_elem_by_elem,
        const char *const SIXTRL_RESTRICT config_str )
    {
        TrackJobBaseNew* ptr_track_job =
            st::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            if( !ptr_track_job->reset( particles_buffer, num_psets,
                    pset_indices_begin, belements_buffer,
                        output_buffer, until_turn_elem_by_elem ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    TrackJobBaseNew* TrackJobNew_create(
        st::arch_id_t const arch_id,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        return st::TrackJobNew_create( arch_id, config_str.c_str() );
    }

    TrackJobBaseNew* TrackJobNew_create(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        std::string const& SIXTRL_RESTRICT_REF conf_str )
    {
        return st::TrackJobNew_create(
            st::ArchInfo_arch_string_to_arch_id( arch_str ), conf_str.c_str());
    }

    TrackJobBaseNew* TrackJobNew_new(
        st::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF belements_buffer,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        TrackJobBaseNew* ptr_track_job =
            st::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            if( !ptr_track_job->reset( particles_buffer, belements_buffer ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }

    TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF belements_buffer,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        return st::TrackJobNew_new(
            st::ArchInfo_arch_string_to_arch_id( arch_str ), particles_buffer,
                belements_buffer, config_str );
    }

    TrackJobBaseNew* TrackJobNew_new(
        st::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer& SIXTRL_RESTRICT_REF belements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        TrackJobBaseNew* ptr_track_job =
            st::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            if( !ptr_track_job->reset( particles_buffer, belements_buffer,
                    output_buffer, until_turn_elem_by_elem ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }

    TrackJobBaseNew* TrackJobNew_new(
        std::string const& SIXTRL_RESTRICT_REF arch_str,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_psets,
        Buffer::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        Buffer& SIXTRL_RESTRICT_REF belements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        return st::TrackJobNew_new(
            st::ArchInfo_arch_string_to_arch_id( arch_str ), particles_buffer,
                num_psets, pset_indices_begin, belements_buffer, output_buffer,
                    until_turn_elem_by_elem, config_str );
    }

    TrackJobBaseNew* TrackJobNew_new(
        st::arch_id_t const arch_id,
        Buffer& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer::size_type const num_psets,
        Buffer::size_type const* SIXTRL_RESTRICT pset_indices_begin,
        Buffer& SIXTRL_RESTRICT_REF belements_buffer,
        Buffer* SIXTRL_RESTRICT output_buffer,
        Buffer::size_type const until_turn_elem_by_elem,
        std::string const& SIXTRL_RESTRICT_REF config_str )
    {
        using size_t = _this_t::size_type;

        TrackJobBaseNew* ptr_track_job =
            st::TrackJobNew_create( arch_id, config_str );

        if( ptr_track_job != nullptr )
        {
            size_t const* pset_indices_end = pset_indices_begin;

            if( ( pset_indices_end != nullptr ) &&
                ( num_psets > size_t{ 0 } ) )
            {
                std::advance( pset_indices_end, num_psets );
            }

            if( !ptr_track_job->reset( particles_buffer, pset_indices_begin,
                    pset_indices_end, belements_buffer, output_buffer,
                        until_turn_elem_by_elem ) )
            {
                delete ptr_track_job;
                ptr_track_job = nullptr;
            }
        }

        return ptr_track_job;
    }
}

#endif /* C++,  Host */

/* end: sixtracklib/common/track/track_job_base.cpp */
