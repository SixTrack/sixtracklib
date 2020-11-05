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
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
    #include "sixtracklib/common/buffer/assign_address_item_kernel_impl.h"
    #include "sixtracklib/common/buffer.hpp"
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

namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        namespace st            = SIXTRL_CXX_NAMESPACE;
        using tjob_t            = st::TrackJobBaseNew;
        using st_size_t         = tjob_t::size_type;
        using st_collect_flag_t = tjob_t::collect_flag_t;
        using st_push_flag_t    = tjob_t::push_flag_t;
        using st_status_t       = tjob_t::status_t;
    }

    st_size_t tjob_t::DefaultNumParticleSetIndices() SIXTRL_NOEXCEPT
    {
        return st::TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS;
    }

    st_size_t const* tjob_t::DefaultParticleSetIndicesBegin() SIXTRL_NOEXCEPT
    {
        return &st::TRACK_JOB_DEFAULT_PARTICLE_SET_INDICES[ 0 ];
    }

    st_size_t const* tjob_t::DefaultParticleSetIndicesEnd() SIXTRL_NOEXCEPT
    {
        st_size_t const* end_ptr = tjob_t::DefaultParticleSetIndicesBegin();
        std::advance( end_ptr, st::TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS );
        return end_ptr;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::clear()
    {
        this->doClear( this->doGetDefaultAllClearFlags() );
    }

    /* --------------------------------------------------------------------- */

    st_collect_flag_t tjob_t::collect()
    {
        return this->doCollect( this->m_collect_flags );
    }

    st_collect_flag_t tjob_t::collect( st_collect_flag_t const flags )
    {
        return this->doCollect( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    st_status_t tjob_t::collectParticles()
    {
        return ( tjob_t::IsCollectFlagSet( this->doCollect(
            st::TRACK_JOB_IO_PARTICLES ), st::TRACK_JOB_IO_PARTICLES ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::collectBeamElements()
    {
        return ( tjob_t::IsCollectFlagSet( this->doCollect(
            st::TRACK_JOB_IO_BEAM_ELEMENTS ), st::TRACK_JOB_IO_BEAM_ELEMENTS ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::collectOutput()
    {
        return ( tjob_t::IsCollectFlagSet( this->doCollect(
            st::TRACK_JOB_IO_OUTPUT ), st::TRACK_JOB_IO_OUTPUT ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::collectDebugFlag()
    {
        return ( tjob_t::IsCollectFlagSet( this->doCollect(
            st::TRACK_JOB_IO_DEBUG_REGISTER ), st::TRACK_JOB_IO_DEBUG_REGISTER ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::collectParticlesAddresses()
    {
        return ( tjob_t::IsCollectFlagSet( this->doCollect(
            st::TRACK_JOB_IO_PARTICLES_ADDR ), st::TRACK_JOB_IO_PARTICLES_ADDR ) )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    void tjob_t::enableCollectParticles()  SIXTRL_NOEXCEPT
    {
        this->m_collect_flags |= st::TRACK_JOB_IO_PARTICLES;
    }

    void tjob_t::disableCollectParticles() SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = tjob_t::UnsetCollectFlag(
            this->m_collect_flags, st::TRACK_JOB_IO_PARTICLES );
    }

    bool tjob_t::isCollectingParticles() const SIXTRL_NOEXCEPT
    {
        return tjob_t::IsCollectFlagSet( this->m_collect_flags,
            st::TRACK_JOB_IO_PARTICLES );
    }

    void tjob_t::enableCollectBeamElements()  SIXTRL_NOEXCEPT
    {
        this->m_collect_flags |= st::TRACK_JOB_IO_BEAM_ELEMENTS;
    }

    void tjob_t::disableCollectBeamElements() SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = tjob_t::UnsetCollectFlag(
            this->m_collect_flags, st::TRACK_JOB_IO_BEAM_ELEMENTS );
    }

    bool tjob_t::isCollectingBeamElements() const SIXTRL_NOEXCEPT
    {
        return tjob_t::IsCollectFlagSet( this->m_collect_flags,
                st::TRACK_JOB_IO_BEAM_ELEMENTS );
    }

    void tjob_t::enableCollectOutput()  SIXTRL_NOEXCEPT
    {
        this->m_collect_flags |= st::TRACK_JOB_IO_OUTPUT;
    }

    void tjob_t::disableCollectOutput() SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = tjob_t::UnsetCollectFlag(
            this->m_collect_flags, st::TRACK_JOB_IO_OUTPUT );
    }

    bool tjob_t::isCollectingOutput() const SIXTRL_NOEXCEPT
    {
        return tjob_t::IsCollectFlagSet( this->m_collect_flags,
                st::TRACK_JOB_IO_OUTPUT );
    }

    st_collect_flag_t tjob_t::collectFlags() const SIXTRL_NOEXCEPT
    {
        return this->m_collect_flags;
    }

    void tjob_t::setCollectFlags( st_collect_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = ( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    bool tjob_t::requiresCollecting() const SIXTRL_NOEXCEPT
    {
        return this->m_requires_collect;
    }

    /* --------------------------------------------------------------------- */

    st_push_flag_t tjob_t::push( st_push_flag_t const push_flag )
    {
        return this->doPush( push_flag & st::TRACK_JOB_PUSH_ALL );
    }

    st_status_t tjob_t::pushParticles()
    {
        return ( ( st::TRACK_JOB_IO_PARTICLES & this->push(
            st::TRACK_JOB_IO_PARTICLES ) ) == st::TRACK_JOB_IO_PARTICLES )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::pushBeamElements()
    {
        return ( ( st::TRACK_JOB_IO_BEAM_ELEMENTS & this->push(
            st::TRACK_JOB_IO_BEAM_ELEMENTS ) ) == st::TRACK_JOB_IO_BEAM_ELEMENTS )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::pushOutput()
    {
        return ( ( st::TRACK_JOB_IO_OUTPUT & this->push(
            st::TRACK_JOB_IO_OUTPUT ) ) == st::TRACK_JOB_IO_OUTPUT )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::fetchParticleAddresses()
    {
        st_status_t status = this->doFetchParticleAddresses();

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

    st_status_t tjob_t::clearParticleAddresses( st_size_t const index )
    {
        return this->doClearParticleAddresses( index );
    }

    st_status_t tjob_t::clearAllParticleAddresses()
    {
        return this->doClearAllParticleAddresses();
    }

    bool tjob_t::canFetchParticleAddresses() const SIXTRL_NOEXCEPT
    {
        return ( this->doGetPtrParticlesAddrBuffer() != nullptr );
    }

    bool tjob_t::hasParticleAddresses() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( !this->m_has_particle_addresses ) ||
            ( ( this->m_has_particle_addresses ) &&
              ( this->canFetchParticleAddresses() ) ) );

        return this->m_has_particle_addresses;
    }

    tjob_t::particles_addr_t const* tjob_t::particleAddresses(
        st_size_t const index ) const SIXTRL_NOEXCEPT
    {
        using ptr_paddr_t = tjob_t::particles_addr_t const*;
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

    tjob_t::buffer_t const*
    tjob_t::ptrParticleAddressesBuffer() const SIXTRL_NOEXCEPT
    {
        return this->doGetPtrParticlesAddrBuffer();
    }

    tjob_t::c_buffer_t const*
    tjob_t::ptrCParticleAddressesBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->doGetPtrParticlesAddrBuffer() != nullptr )
            ? this->doGetPtrParticlesAddrBuffer()->getCApiPtr() : nullptr;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::track_status_t tjob_t::trackUntil( st_size_t const until_turn )
    {
        return this->doTrackUntilTurn( until_turn );
    }

    tjob_t::track_status_t tjob_t::trackElemByElem(
        st_size_t const until_turn_elem_by_elem )
    {
        return this->doTrackElemByElem( until_turn_elem_by_elem );
    }

    tjob_t::track_status_t tjob_t::trackLine(
        st_size_t const be_begin_index, st_size_t const be_end_index,
        bool const finish_turn )
    {
        return this->doTrackLine( be_begin_index, be_end_index, finish_turn );
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::reset(
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF be_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem  )
    {
        using c_buffer_t = tjob_t::c_buffer_t;
        using clear_flag_t = tjob_t::clear_flag_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

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

    st_status_t tjob_t::reset(
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        st_size_t const pset_index,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem  )
    {
        tjob_t::c_buffer_t* ptr_out = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        st_status_t status = tjob_t::reset(
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

    st_status_t tjob_t::reset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem  )
    {
        using clear_flag_t = tjob_t::clear_flag_t;

        clear_flag_t const clear_flags = this->doPrepareResetClearFlags(
            particles_buffer, this->numParticleSets(),
                this->particleSetIndicesBegin(), be_buffer, ptr_output_buffer,
                    until_turn_elem_by_elem );

        this->doClear( clear_flags );

        return this->doReset( particles_buffer, be_buffer, ptr_output_buffer,
            until_turn_elem_by_elem );
    }

    st_status_t tjob_t::reset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        st_size_t const particle_set_index,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem  )
    {
        return tjob_t::reset( particles_buffer,
            st_size_t{ 1 }, &particle_set_index, be_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
    }

    st_status_t tjob_t::reset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        st_size_t const num_particle_sets,
        st_size_t const* SIXTRL_RESTRICT pset_indices_begin,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem  )
    {
        using size_t = st_size_t;

        tjob_t::clear_flag_t const clear_flags =
            this->doPrepareResetClearFlags( particles_buffer,
                num_particle_sets, pset_indices_begin, beam_elements_buffer,
                    ptr_output_buffer, until_turn_elem_by_elem );

        this->doClear( clear_flags );

        st_status_t status = st::ARCH_STATUS_SUCCESS;

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
    st_status_t tjob_t::selectParticleSets(
        st_size_t const num_particle_sets,
        st_size_t const*
            SIXTRL_RESTRICT particle_set_indices_begin )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->ptrCParticlesBuffer() != nullptr )
        {


        }

        return status;
    }
    */

    st_status_t tjob_t::selectParticleSet(
        st_size_t const particle_set_index )
    {
        using buffer_t   = tjob_t::buffer_t;
        using c_buffer_t = tjob_t::c_buffer_t;
        using size_t = st_size_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

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

    st_status_t tjob_t::assignOutputBuffer(
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF output_buffer )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

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

    st_status_t tjob_t::assignOutputBuffer(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        return this->doAssignNewOutputBuffer( ptr_output_buffer );
    }

    /* --------------------------------------------------------------------- */

    st_size_t tjob_t::numParticleSets() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_indices.size();
    }

    st_size_t const*
    tjob_t::particleSetIndicesBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_indices.data();
    }

    st_size_t const*
    tjob_t::particleSetIndicesEnd() const SIXTRL_NOEXCEPT
    {
        st_size_t const* ptr =
            this->particleSetIndicesBegin();

        SIXTRL_ASSERT( ptr != nullptr );
        std::advance( ptr, this->numParticleSets() );
        return ptr;
    }

    st_size_t tjob_t::particleSetIndex(
        st_size_t const idx ) const
    {
        return this->m_particle_set_indices.at( idx );
    }

    /* --------------------------------------------------------------------- */

    tjob_t::num_particles_t tjob_t::particleSetBeginIndexOffset(
        st_size_t const pset_index ) const
    {
        return this->m_particle_set_begin_offsets.at( pset_index );
    }

    tjob_t::num_particles_t tjob_t::particleSetEndIndexOffset(
        st_size_t const pset_index ) const
    {
        return this->m_particle_set_end_offsets.at( pset_index );
    }

    tjob_t::num_particles_t tjob_t::numParticlesInSet(
        st_size_t const pset_index ) const SIXTRL_NOEXCEPT
    {
        using num_particles_t = tjob_t::num_particles_t;
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

    tjob_t::num_particles_t const*
    tjob_t::particleSetBeginOffsetsBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_begin_offsets.data();
    }

    tjob_t::num_particles_t const*
    tjob_t::particleSetBeginOffsetsEnd() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->numParticleSets() ==
                       this->m_particle_set_begin_offsets.size() );

        tjob_t::num_particles_t const* end_ptr =
            this->particleSetBeginOffsetsBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numParticleSets() );
        }

        return end_ptr;
    }

    tjob_t::num_particles_t const*
    tjob_t::particleSetEndOffsetsBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_end_offsets.data();
    }

    tjob_t::num_particles_t const*
    tjob_t::particleSetEndOffsetsEnd() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->numParticleSets() ==
                       this->m_particle_set_end_offsets.size() );

        tjob_t::num_particles_t const* end_ptr =
            this->particleSetEndOffsetsBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numParticleSets() );
        }

        return end_ptr;
    }

    tjob_t::num_particles_t
    tjob_t::totalNumParticles() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles;
    }

    tjob_t::num_particles_t
    tjob_t::totalNumParticlesInParticleSets() const SIXTRL_NOEXCEPT
    {
        return this->m_total_num_particles_in_sets;
    }

    st_size_t
    tjob_t::totalNumOfAvailableParticleSets() const SIXTRL_NOEXCEPT
    {
        return this->m_num_particle_sets_in_buffer;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::particle_index_t
    tjob_t::minParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_particle_id;
    }

    tjob_t::particle_index_t
    tjob_t::maxParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_particle_id;
    }

    tjob_t::particle_index_t
    tjob_t::minElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_min_element_id;
    }

    tjob_t::particle_index_t
    tjob_t::maxElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_max_element_id;
    }

    tjob_t::particle_index_t
    tjob_t::minInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_initial_turn_id;
    }

    tjob_t::particle_index_t
    tjob_t::maxInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_initial_turn_id;
    }

    /* --------------------------------------------------------------------- */

    st_size_t
    tjob_t::totalNumOfAvailableBeamElements() const SIXTRL_NOEXCEPT
    {
        return this->m_num_beam_elements_in_buffer;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::buffer_t*
    tjob_t::ptrParticlesBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::tjob_t::buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrParticlesBuffer() );
    }

    tjob_t::buffer_t const*
    tjob_t::ptrParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_particles_buffer;
    }

    tjob_t::c_buffer_t*
    tjob_t::ptrCParticlesBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::tjob_t::c_buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrCParticlesBuffer() );
    }

    tjob_t::c_buffer_t const*
    tjob_t::ptrCParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_c_particles_buffer;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::buffer_t*
    tjob_t::ptrBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::tjob_t::buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrBeamElementsBuffer() );
    }

    tjob_t::buffer_t const*
    tjob_t::ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_beam_elem_buffer;
    }

    tjob_t::c_buffer_t*
    tjob_t::ptrCBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::tjob_t::c_buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrCBeamElementsBuffer() );
    }

    tjob_t::c_buffer_t const*
    tjob_t::ptrCBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_c_beam_elem_buffer;
    }

    /* ----------------------------------------------------------------- */

    tjob_t::assign_item_t* tjob_t::add_assign_address_item(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF assign_item_to_add )
    {
        tjob_t::assign_item_t* item = nullptr;

        if( assign_item_to_add.valid() )
        {
            tjob_t::assign_item_key_t const key =
            tjob_t::assign_item_key_t{
                assign_item_to_add.getDestBufferId(),
                assign_item_to_add.getSrcBufferId() };

            st_size_t item_index = std::numeric_limits< st_size_t >::max();
            st_status_t const status = this->do_add_assign_address_item(
                assign_item_to_add, &item_index );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                item = this->do_get_assign_address_item( key, item_index );
            }
        }

        return item;
    }

    tjob_t::assign_item_t* tjob_t::add_assign_address_item(
        tjob_t::object_type_id_t const dest_type_id,
        st_size_t const dest_buffer_id,
        st_size_t const dest_elem_index,
        st_size_t const dest_pointer_offset,
        tjob_t::object_type_id_t const src_type_id,
        st_size_t const src_buffer_id,
        st_size_t const src_elem_index,
        st_size_t const src_pointer_offset )
    {
        return this->add_assign_address_item( tjob_t::assign_item_t{
            dest_type_id, dest_buffer_id, dest_elem_index, dest_pointer_offset,
            src_type_id, src_buffer_id, src_elem_index, src_pointer_offset } );
    }

    st_status_t tjob_t::remove_assign_address_item(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF item_to_remove )
    {
        using key_t = tjob_t::assign_item_key_t;
        key_t const key = key_t { item_to_remove.getDestBufferId(),
                                  item_to_remove.getSrcBufferId() };
        st_size_t const item_index = this->do_find_assign_address_item(
            item_to_remove );
        return this->do_remove_assign_address_item( key, item_index );
    }

    st_status_t tjob_t::remove_assign_address_item(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key,
        st_size_t const index_of_item_to_remove )
    {
        return this->do_remove_assign_address_item(
            key, index_of_item_to_remove );
    }

    bool tjob_t::has_assign_address_item( tjob_t::assign_item_t const&
        SIXTRL_RESTRICT_REF assign_item ) const SIXTRL_NOEXCEPT
    {
        st_size_t const item_index = this->do_find_assign_address_item(
            assign_item );

        return ( item_index < this->num_assign_items(
            assign_item.getDestBufferId(), assign_item.getSrcBufferId() ) );
    }

    bool tjob_t::has_assign_address_item(
        tjob_t::object_type_id_t const dest_type_id,
        st_size_t const dest_buffer_id,
        st_size_t const dest_elem_index,
        st_size_t const dest_pointer_offset,
        tjob_t::object_type_id_t const src_type_id,
        st_size_t const src_buffer_id,
        st_size_t const src_elem_index,
        st_size_t const src_pointer_offset ) const SIXTRL_NOEXCEPT
    {
        return this->has_assign_address_item( tjob_t::assign_item_t{
            dest_type_id, dest_buffer_id, dest_elem_index, dest_pointer_offset,
            src_type_id, src_buffer_id, src_elem_index, src_pointer_offset } );
    }

    st_size_t tjob_t::index_of_assign_address_item( tjob_t::assign_item_t
        const& SIXTRL_RESTRICT_REF assign_item ) const SIXTRL_NOEXCEPT
    {
        return this->do_find_assign_address_item( assign_item );
    }

    st_size_t tjob_t::index_of_assign_address_item(
        tjob_t::object_type_id_t const dest_type_id,
        st_size_t const dest_buffer_id,
        st_size_t const dest_elem_index,
        st_size_t const dest_pointer_offset,
        tjob_t::object_type_id_t const src_type_id,
        st_size_t const src_buffer_id,
        st_size_t const src_elem_index,
        st_size_t const src_pointer_offset ) const SIXTRL_NOEXCEPT
    {
        return this->do_find_assign_address_item( tjob_t::assign_item_t{
            dest_type_id, dest_buffer_id, dest_elem_index, dest_pointer_offset,
            src_type_id, src_buffer_id, src_elem_index, src_pointer_offset } );
    }

    bool tjob_t::has_assign_items( st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id ) const SIXTRL_NOEXCEPT
    {
        return ( this->num_assign_items( dest_buffer_id, src_buffer_id ) >
                    st_size_t{ 0 } );
    }

    st_size_t tjob_t::num_assign_items( st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id ) const SIXTRL_NOEXCEPT
    {
        st_size_t num_items = st_size_t{ 0 };

        auto it = this->m_assign_address_items.find(
            tjob_t::assign_item_key_t{ dest_buffer_id, src_buffer_id } );

        if( it != this->m_assign_address_items.end() )
        {
            num_items = it->second.getNumObjects();
        }

        return num_items;
    }

    st_size_t tjob_t::num_distinct_available_assign_address_items_dest_src_pairs()
        const SIXTRL_NOEXCEPT
    {
        return this->m_assign_item_keys.size();
    }

    st_size_t tjob_t::available_assign_address_items_dest_src_pairs(
        st_size_t const max_num_pairs,
        tjob_t::assign_item_key_t* pairs_begin ) const SIXTRL_NOEXCEPT
    {
        using size_t = st_size_t;
        size_t num_pairs = size_t{ 0 };

        if( ( max_num_pairs > size_t{ 0 } ) && ( pairs_begin != nullptr ) &&
            ( !this->m_assign_address_items.empty() ) )
        {
            tjob_t::assign_item_key_t* pairs_end = pairs_begin;
            std::advance( pairs_end, max_num_pairs );

            num_pairs = this->available_assign_address_items_dest_src_pairs(
                pairs_begin, pairs_end );
        }

        return num_pairs;
    }

    tjob_t::c_buffer_t* tjob_t::buffer_by_buffer_id(
        st_size_t const buffer_id ) SIXTRL_NOEXCEPT
    {
        using ptr_t = tjob_t::c_buffer_t*;
        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).buffer_by_buffer_id( buffer_id ) );
    }

    tjob_t::c_buffer_t const* tjob_t::buffer_by_buffer_id(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        tjob_t::c_buffer_t const* ptr_buffer = nullptr;

        switch( buffer_id )
        {
            case st::ARCH_PARTICLES_BUFFER_ID:
            {
                ptr_buffer = this->ptrCParticlesBuffer();
                break;
            }

            case st::ARCH_BEAM_ELEMENTS_BUFFER_ID:
            {
                ptr_buffer = this->ptrCBeamElementsBuffer();
                break;
            }

            case st::ARCH_OUTPUT_BUFFER_ID:
            {
                ptr_buffer = this->ptrCOutputBuffer();
                break;
            }

            case st::ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID:
            {
                if( this->m_my_elem_by_elem_buffer.get() != nullptr )
                {
                    ptr_buffer = this->m_my_elem_by_elem_buffer->getCApiPtr();
                }

                break;
            }

            case st::ARCH_PARTICLE_ADDR_BUFFER_ID:
            {
                if( this->ptrParticleAddressesBuffer() != nullptr )
                {
                    ptr_buffer = this->ptrParticleAddressesBuffer(
                        )->getCApiPtr();
                }
                break;
            }

            default:
            {
                if( ( buffer_id >= st::ARCH_MIN_USER_DEFINED_BUFFER_ID ) &&
                    ( buffer_id <= st::ARCH_MAX_USER_DEFINED_BUFFER_ID ) )
                {
                    ptr_buffer = this->ptr_stored_buffer( buffer_id );
                }
            }
        };

        return ptr_buffer;
    }

    bool tjob_t::is_buffer_by_buffer_id(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( buffer_id == st::ARCH_PARTICLES_BUFFER_ID ) ||
                 ( buffer_id == st::ARCH_BEAM_ELEMENTS_BUFFER_ID ) ||
                 ( buffer_id == st::ARCH_OUTPUT_BUFFER_ID ) ||
                 ( buffer_id == st::ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID ) ||
                 ( buffer_id == st::ARCH_PARTICLE_ADDR_BUFFER_ID ) ||
                 ( ( buffer_id >= st::ARCH_MIN_USER_DEFINED_BUFFER_ID ) &&
                   ( buffer_id <= st::ARCH_MAX_USER_DEFINED_BUFFER_ID ) ) );
    }

    bool tjob_t::is_raw_memory_by_buffer_id(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        return ( ( buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
                 ( !this->is_buffer_by_buffer_id( buffer_id ) ) );
    }

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object) const*
    tjob_t::assign_items_begin( st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id ) const SIXTRL_NOEXCEPT
    {
        auto it = this->m_assign_address_items.find(
            tjob_t::assign_item_key_t{ dest_buffer_id, src_buffer_id } );

        return ( it != this->m_assign_address_items.end() )
            ? ::NS(Buffer_get_const_objects_begin)( it->second.getCApiPtr() )
            : nullptr;
    }

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object) const*
    tjob_t::assign_items_end( st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id ) const SIXTRL_NOEXCEPT
    {
        auto it = this->m_assign_address_items.find(
            tjob_t::assign_item_key_t{ dest_buffer_id, src_buffer_id } );

        return ( it != this->m_assign_address_items.end() )
            ? ::NS(Buffer_get_const_objects_begin)( it->second.getCApiPtr() )
            : nullptr;
    }

    tjob_t::assign_item_key_t const*
    tjob_t::assign_item_dest_src_begin() const SIXTRL_NOEXCEPT
    {
        return this->m_assign_item_keys.data();
    }

    tjob_t::assign_item_key_t const*
    tjob_t::assign_item_dest_src_end() const SIXTRL_NOEXCEPT
    {
        tjob_t::assign_item_key_t const* end_ptr =
            this->assign_item_dest_src_begin();

        if( ( end_ptr != nullptr ) && ( this->m_assign_item_keys.empty() ) )
        {
            std::advance( end_ptr, this->m_assign_address_items.size() );
        }

        return end_ptr;
    }

    st_size_t tjob_t::total_num_assign_items() const SIXTRL_NOEXCEPT
    {
        using size_t = st_size_t;
        size_t total_num_items = size_t{ 0 };

        auto it = this->m_assign_address_items.begin();
        auto end = this->m_assign_address_items.end();

        for( ; it != end ; ++it ) total_num_items += it->second.getNumObjects();
        return total_num_items;
    }

    tjob_t::assign_item_t const* tjob_t::ptr_assign_address_item(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF
            assign_address_item ) const SIXTRL_NOEXCEPT
    {
        st_size_t const assign_item_idx =
            this->do_find_assign_address_item( assign_address_item );

        return this->do_get_assign_address_item( tjob_t::assign_item_key_t{
                assign_address_item.getDestBufferId(),
                assign_address_item.getSrcBufferId() }, assign_item_idx );
    }


    tjob_t::assign_item_t const* tjob_t::ptr_assign_address_item(
        st_size_t const dest_buffer_id, st_size_t const src_buffer_id,
        st_size_t const assign_item_index ) const SIXTRL_NOEXCEPT
    {
        return this->do_get_assign_address_item(
            tjob_t::assign_item_key_t{ dest_buffer_id, src_buffer_id },
                assign_item_index );
    }

    tjob_t::assign_item_t const* tjob_t::ptr_assign_address_item(
        tjob_t::object_type_id_t const dest_type_id,
        st_size_t const dest_buffer_id,
        st_size_t const dest_elem_index,
        st_size_t const dest_pointer_offset,
        tjob_t::object_type_id_t const src_type_id,
        st_size_t const src_buffer_id,
        st_size_t const src_elem_index,
        st_size_t const src_pointer_offset ) const SIXTRL_NOEXCEPT
    {
        return this->ptr_assign_address_item(
            tjob_t::assign_item_t{ dest_type_id, dest_buffer_id,
                dest_elem_index, dest_pointer_offset, src_type_id,
                    src_buffer_id, src_elem_index, src_pointer_offset } );
    }

    st_status_t tjob_t::commit_address_assignments()
    {
        return this->do_commit_address_assignments();
    }

    st_status_t tjob_t::assign_all_addresses()
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;

        auto it = this->m_assign_address_items.begin();
        auto end = this->m_assign_address_items.end();

        for( ; it != end ; ++it )
        {
            status = this->do_perform_address_assignments( it->first );
            if( status != st::ARCH_STATUS_SUCCESS ) break;
        }

        return status;
    }

    st_status_t tjob_t::assign_addresses(
        st_size_t const dest_buffer_id, st_size_t const src_buffer_id )
    {
        return this->do_perform_address_assignments(
            tjob_t::assign_item_key_t{ dest_buffer_id, src_buffer_id } );
    }

    /* ---------------------------------------------------------------- */

    st_size_t tjob_t::stored_buffers_capacity() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_buffers.capacity();
    }

    st_status_t tjob_t::reserve_stored_buffers_capacity(
        st_size_t const capacity )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        if( capacity < this->m_stored_buffers.capacity() )
        {
            this->m_stored_buffers.reserve( capacity );
            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    bool tjob_t::has_stored_buffers() const SIXTRL_NOEXCEPT
    {
        return ( this->num_stored_buffers() > st_size_t{ 0 } );
    }

    st_size_t tjob_t::num_stored_buffers() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( std::count_if( this->m_stored_buffers.begin(),
            this->m_stored_buffers.end(),
            []( tjob_t::buffer_store_t const& s) { return s.active(); } ) ==
            static_cast< std::ptrdiff_t >( this->m_num_stored_buffers ) );

        SIXTRL_ASSERT( this->m_stored_buffers.size() >=
                       this->m_num_stored_buffers );

        return this->m_num_stored_buffers;
    }

    st_size_t tjob_t::min_stored_buffer_id() const SIXTRL_NOEXCEPT
    {
        return ( this->has_stored_buffers() )
            ? st::ARCH_MIN_USER_DEFINED_BUFFER_ID : st::ARCH_ILLEGAL_BUFFER_ID;
    }

    st_size_t tjob_t::max_stored_buffer_id() const SIXTRL_NOEXCEPT
    {
        st_size_t max_stored_buffers_id = this->min_stored_buffer_id();

        if( max_stored_buffers_id != st::ARCH_ILLEGAL_BUFFER_ID )
        {
            SIXTRL_ASSERT( this->m_stored_buffers.size() >= st_size_t{ 1 } );
            max_stored_buffers_id += this->m_stored_buffers.size();
            max_stored_buffers_id -= st_size_t{ 1 };
        }

        return max_stored_buffers_id;
    }

    bool tjob_t::owns_stored_buffer( st_size_t const buffer_id
        ) const SIXTRL_NOEXCEPT
    {
        auto ptr_buffer_store = this->do_get_ptr_buffer_store( buffer_id );
        return ( ( ptr_buffer_store != nullptr ) &&
                 ( ptr_buffer_store->owns_buffer() ) );
    }

    st_status_t tjob_t::remove_stored_buffer( st_size_t const buffer_index )
    {
        return this->do_remove_stored_buffer( buffer_index );
    }

    tjob_t::buffer_t& tjob_t::stored_cxx_buffer( st_size_t const buffer_id )
    {
        return const_cast< tjob_t::buffer_t& >( static_cast< tjob_t const& >(
            *this ).stored_cxx_buffer( buffer_id ) );
    }

    tjob_t::buffer_t const& tjob_t::stored_cxx_buffer(
        st_size_t const buffer_id ) const
    {
        tjob_t::buffer_t const* ptr_buffer =
            this->ptr_stored_cxx_buffer( buffer_id );

        if( ptr_buffer == nullptr )
        {
            throw std::runtime_error(
                "stored buffer does not have a c++ representation" );
        }

        return *ptr_buffer;
    }

    tjob_t::buffer_t* tjob_t::ptr_stored_cxx_buffer(
        st_size_t const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::buffer_t* >( static_cast< tjob_t const& >(
            *this ).ptr_stored_cxx_buffer( buffer_id ) );
    }

    tjob_t::buffer_t const* tjob_t::ptr_stored_cxx_buffer(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        tjob_t::buffer_store_t const* ptr_stored_buffer =
            this->do_get_ptr_buffer_store( buffer_id );

        return ( ptr_stored_buffer != nullptr )
            ? ptr_stored_buffer->ptr_cxx_buffer() : nullptr;
    }

    tjob_t::c_buffer_t* tjob_t::ptr_stored_buffer(
        st_size_t const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::c_buffer_t* >( static_cast<
            tjob_t const& >( *this ).ptr_stored_buffer( buffer_id ) );
    }

    tjob_t::c_buffer_t const* tjob_t::ptr_stored_buffer(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        tjob_t::buffer_store_t const* ptr_stored_buffer =
            this->do_get_ptr_buffer_store( buffer_id );

        return ( ptr_stored_buffer != nullptr )
            ? ptr_stored_buffer->ptr_buffer() : nullptr;
    }

    st_status_t tjob_t::push_stored_buffer(
        st_size_t const buffer_id )
    {
        return this->do_push_stored_buffer( buffer_id );
    }

    st_status_t tjob_t::collect_stored_buffer(
        st_size_t const buffer_id )
    {
        return this->do_collect_stored_buffer( buffer_id );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasOutputBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCOutputBuffer() != nullptr );
    }

    bool tjob_t::ownsOutputBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_my_output_buffer.get() == nullptr ) ||
            ( ( this->m_my_output_buffer.get() ==
                this->m_ptr_output_buffer ) &&
              ( this->m_my_output_buffer->getCApiPtr() ==
                this->m_ptr_c_output_buffer ) ) );

        return ( ( this->ptrOutputBuffer() != nullptr ) &&
                 ( this->m_my_output_buffer.get() != nullptr ) );
    }

    bool tjob_t::hasElemByElemOutput() const SIXTRL_NOEXCEPT
    {
        return this->m_has_elem_by_elem_output;
    }

    bool tjob_t::hasBeamMonitorOutput() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( (  !this->m_has_beam_monitor_output ) ||
            ( ( this->m_has_beam_monitor_output ) &&
              ( this->m_ptr_c_output_buffer != nullptr ) ) );

        return this->m_has_beam_monitor_output;
    }

    st_size_t
    tjob_t::beamMonitorsOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_be_mon_output_buffer_offset ) ) ||
            ( this->m_be_mon_output_buffer_offset ==
                st_size_t{ 0 } ) );

        return this->m_be_mon_output_buffer_offset;
    }

    st_size_t
    tjob_t::elemByElemOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_elem_by_elem_output_offset ) ) ||
            ( this->m_elem_by_elem_output_offset ==
                st_size_t{ 0 } ) );

        return this->m_elem_by_elem_output_offset;
    }

    tjob_t::particle_index_t
    tjob_t::untilTurnElemByElem() const SIXTRL_NOEXCEPT
    {
        return this->m_until_turn_elem_by_elem;
    }

    st_size_t
    tjob_t::numElemByElemTurns() const SIXTRL_NOEXCEPT
    {
        if( ( this->m_until_turn_elem_by_elem > this->m_min_initial_turn_id ) &&
            ( this->m_min_initial_turn_id >= tjob_t::particle_index_t{ 0 } ) )
        {
            return static_cast< size_t >( this->m_until_turn_elem_by_elem -
                this->m_min_initial_turn_id );
        }

        return st_size_t{ 0 };
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    tjob_t::buffer_t* tjob_t::ptrOutputBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::tjob_t::buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrOutputBuffer() );
    }

    tjob_t::buffer_t const*
    tjob_t::ptrOutputBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_output_buffer == nullptr ) ||
            ( this->m_ptr_output_buffer->getCApiPtr() ==
              this->m_ptr_c_output_buffer ) );

        return this->m_ptr_output_buffer;
    }

    tjob_t::c_buffer_t* tjob_t::ptrCOutputBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< st::tjob_t::c_buffer_t* >( static_cast<
            st::TrackJobBaseNew const& >( *this ).ptrCOutputBuffer() );
    }

    tjob_t::c_buffer_t const*
    tjob_t::ptrCOutputBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_output_buffer == nullptr ) ||
            ( this->m_ptr_output_buffer->getCApiPtr() ==
              this->m_ptr_c_output_buffer ) );

        return this->m_ptr_c_output_buffer;
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasBeamMonitors() const SIXTRL_NOEXCEPT
    {
        return !this->m_beam_monitor_indices.empty();
    }

    st_size_t tjob_t::numBeamMonitors() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_indices.size();
    }

    st_size_t const*
    tjob_t::beamMonitorIndicesBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_indices.data();
    }

    st_size_t const*
    tjob_t::beamMonitorIndicesEnd() const SIXTRL_NOEXCEPT
    {
        st_size_t const* ptr = this->beamMonitorIndicesBegin();
        if( ptr != nullptr ) std::advance( ptr, this->numBeamMonitors() );
        return ptr;
    }

    st_size_t tjob_t::beamMonitorIndex(
        st_size_t const idx ) const
    {
        return this->m_beam_monitor_indices.at( idx );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrElemByElemConfig() != nullptr );
    }

    tjob_t::elem_by_elem_config_t const*
    tjob_t::ptrElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_my_elem_by_elem_buffer.get() != nullptr );
        return ::NS(ElemByElemConfig_const_from_buffer)(
            this->m_my_elem_by_elem_buffer->getCApiPtr(),
                this->m_elem_by_elem_config_index );
    }

    tjob_t::elem_by_elem_config_t*
    tjob_t::ptrElemByElemConfig() SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_my_elem_by_elem_buffer.get() != nullptr );
        return ::NS(ElemByElemConfig_from_buffer)(
            this->m_my_elem_by_elem_buffer->getCApiPtr(),
                this->m_elem_by_elem_config_index );
    }

    bool tjob_t::elemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return ::NS(ElemByElemConfig_is_rolling)( this->ptrElemByElemConfig() );
    }

    bool tjob_t::defaultElemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return this->m_default_elem_by_elem_rolling;
    }

    void tjob_t::setDefaultElemByElemRolling(
        bool const is_rolling ) SIXTRL_NOEXCEPT
    {
        this->m_default_elem_by_elem_rolling = is_rolling;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::clear_flag_t
    tjob_t::doGetDefaultAllClearFlags() const SIXTRL_NOEXCEPT
    {
        return this->m_clear_all_flags;
    }

    void tjob_t::doSetDefaultAllClearFlags(
        tjob_t::clear_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        this->m_clear_all_flags = flags;
    }

    tjob_t::clear_flag_t
    tjob_t::doGetDefaultPrepareResetClearFlags() const SIXTRL_NOEXCEPT
    {
        return this->m_clear_prepare_reset_flags;
    }

    void tjob_t::doSetDefaultPrepareResetClearFlags(
        tjob_t::clear_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        this->m_clear_prepare_reset_flags = flags;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    tjob_t::buffer_store_t const* tjob_t::do_get_ptr_buffer_store(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        tjob_t::buffer_store_t const* ptr_buffer_store = nullptr;
        st_size_t const min_buffer_id = this->min_stored_buffer_id();
        st_size_t const max_buffer_id_plus_one =
            min_buffer_id + this->m_stored_buffers.size();

        if( ( min_buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_id >= min_buffer_id ) &&
            ( buffer_id <  max_buffer_id_plus_one ) )
        {
            SIXTRL_ASSERT( this->do_get_stored_buffer_size() >=
                           this->m_num_stored_buffers );

            ptr_buffer_store = &this->m_stored_buffers[
                buffer_id - min_buffer_id ];
        }

        return ptr_buffer_store;
    }

    tjob_t::buffer_store_t* tjob_t::do_get_ptr_buffer_store(
        st_size_t const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::buffer_store_t* >( static_cast<
            tjob_t const& >( *this ).do_get_ptr_buffer_store( buffer_id ) );
    }

    st_size_t tjob_t::do_get_stored_buffer_size() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_buffers.size();
    }

    st_size_t tjob_t::do_find_assign_address_item( tjob_t::assign_item_t const&
        SIXTRL_RESTRICT_REF item_to_search ) const SIXTRL_NOEXCEPT
    {
        using item_t = tjob_t::assign_item_t;
        using key_t  = tjob_t::assign_item_key_t;

        size_t index = std::numeric_limits< size_t >::max();

        key_t const key = key_t{ item_to_search.getDestBufferId(),
                                 item_to_search.getSrcBufferId() };

        auto it = this->m_assign_address_items.find( key );

        SIXTRL_ASSERT( this->m_assign_address_items.size() ==
                       this->m_assign_item_keys.size() );

        if( it != this->m_assign_address_items.end() )
        {
            auto cmp_key_fn = []( key_t const& SIXTRL_RESTRICT_REF lhs,
                                  key_t const& SIXTRL_RESTRICT_REF rhs )
            {
                return ( ( lhs.dest_buffer_id < rhs.dest_buffer_id ) ||
                         ( ( lhs.dest_buffer_id == rhs.dest_buffer_id ) &&
                           ( lhs.src_buffer_id < rhs.src_buffer_id ) ) );
            };

            SIXTRL_ASSERT( std::is_sorted( this->m_assign_item_keys.begin(),
                this->m_assign_item_keys.end(), cmp_key_fn ) );

            SIXTRL_ASSERT( std::binary_search( this->m_assign_item_keys.begin(),
                this->m_assign_item_keys.end(), key, cmp_key_fn ) );

            ( void )cmp_key_fn;

            size_t const nn = it->second.getNumObjects();
            ::NS(AssignAddressItem) const* ptr_item_to_search =
                item_to_search.getCApiPtr();

            for( size_t ii = size_t{ 0 } ; ii < nn ; ++ii )
            {
                tjob_t::assign_item_t const* item =
                    it->second.get< item_t >( ii );

                if( ( item != nullptr ) &&
                    ( ::NS(AssignAddressItem_are_equal)(
                        item->getCApiPtr(), ptr_item_to_search ) ) )
                {
                    index = ii;
                    break;
                }
            }

            if( index > nn ) index = nn;
        }

        return index;
    }

    tjob_t::assign_item_t const* tjob_t::do_get_assign_address_item(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key,
        st_size_t const item_index ) const SIXTRL_NOEXCEPT
    {
        using item_t = tjob_t::assign_item_t;
        item_t const* ptr_found_item = nullptr;

        auto it = this->m_assign_address_items.find( key );

        if( ( it != this->m_assign_address_items.end() ) &&
            ( it->second.getNumObjects() > item_index ) )
        {
            ptr_found_item = it->second.get< item_t >( item_index );
        }

        return ptr_found_item;
    }

    tjob_t::assign_item_t* tjob_t::do_get_assign_address_item(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key,
        st_size_t const item_index ) SIXTRL_NOEXCEPT
    {
        using ptr_t = tjob_t::assign_item_t*;
        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).do_get_assign_address_item( key, item_index ) );
    }

    tjob_t::c_buffer_t* tjob_t::do_get_ptr_assign_address_items_buffer(
        tjob_t::assign_item_key_t const&
            SIXTRL_RESTRICT_REF key ) SIXTRL_NOEXCEPT
    {
        tjob_t::c_buffer_t* ptr_assign_address_items_buffer = nullptr;
        auto it = this->m_assign_address_items.find( key );

        if( it != this->m_assign_address_items.end() )
        {
            ptr_assign_address_items_buffer = it->second.getCApiPtr();
        }

        return ptr_assign_address_items_buffer;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::buffer_t const& tjob_t::elem_by_elem_config_cxx_buffer() const
    {
        if( this->m_my_elem_by_elem_buffer.get() == nullptr )
        {
            throw std::runtime_error( "no elem-by-elem config buffer available" );
        }

        return *( this->m_my_elem_by_elem_buffer.get() );
    }

    tjob_t::buffer_t& tjob_t::elem_by_elem_config_cxx_buffer()
    {
        return const_cast< tjob_t::buffer_t& >( static_cast< tjob_t const& >(
            *this ).elem_by_elem_config_cxx_buffer() );
    }

    tjob_t::c_buffer_t const*
    tjob_t::elem_by_elem_config_buffer() const SIXTRL_NOEXCEPT
    {
        return ( this->m_my_elem_by_elem_buffer.get() != nullptr )
            ? this->m_my_elem_by_elem_buffer->getCApiPtr() : nullptr;
    }

    tjob_t::c_buffer_t* tjob_t::elem_by_elem_config_buffer() SIXTRL_NOEXCEPT
    {
        return ( this->m_my_elem_by_elem_buffer.get() != nullptr )
            ? this->m_my_elem_by_elem_buffer->getCApiPtr() : nullptr;
    }

    st_size_t tjob_t::elem_by_elem_config_index() const SIXTRL_NOEXCEPT
    {
        return this->m_elem_by_elem_config_index;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::elem_by_elem_order_t tjob_t::elemByElemOrder() const SIXTRL_NOEXCEPT
    {
        return ::NS(ElemByElemConfig_get_order)( this->ptrElemByElemConfig() );
    }

    tjob_t::elem_by_elem_order_t
    tjob_t::defaultElemByElemOrder() const SIXTRL_NOEXCEPT
    {
        return this->m_default_elem_by_elem_order;
    }

    void tjob_t::setDefaultElemByElemOrder(
        tjob_t::elem_by_elem_order_t const order ) SIXTRL_NOEXCEPT
    {
        this->m_default_elem_by_elem_order = order;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::TrackJobBaseNew(
        tjob_t::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str ) :
        st::ArchDebugBase( arch_id, arch_str, config_str ),
        m_assign_address_items(),
        m_particle_set_indices(),
        m_particle_set_begin_offsets(),
        m_particle_set_end_offsets(),
        m_beam_monitor_indices(),
        m_stored_buffers(),
        m_assign_item_keys(),
        m_my_output_buffer( nullptr ),
        m_my_elem_by_elem_buffer( new tjob_t::buffer_t ),
        m_my_particles_addr_buffer( new tjob_t::buffer_t ),
        m_ptr_particles_buffer( nullptr ),
        m_ptr_beam_elem_buffer( nullptr ), m_ptr_output_buffer( nullptr ),
        m_ptr_c_particles_buffer( nullptr ),
        m_ptr_c_beam_elem_buffer( nullptr ), m_ptr_c_output_buffer( nullptr ),
        m_be_mon_output_buffer_offset( st_size_t{ 0 } ),
        m_elem_by_elem_output_offset( st_size_t{ 0 } ),
        m_num_particle_sets_in_buffer( st_size_t{ 0 } ),
        m_num_beam_elements_in_buffer( st_size_t{ 0 } ),
        m_elem_by_elem_config_index( st_size_t{ 0 } ),
        m_num_stored_buffers( st_size_t{ 0 } ),
        m_default_elem_by_elem_order( ::NS(ELEM_BY_ELEM_ORDER_DEFAULT) ),
        m_total_num_particles( tjob_t::num_particles_t{ 0 } ),
        m_total_num_particles_in_sets( tjob_t::num_particles_t{ 0 } ),
        m_min_particle_id( tjob_t::particle_index_t{ 0 } ),
        m_max_particle_id( tjob_t::particle_index_t{ 0 } ),
        m_min_element_id( tjob_t::particle_index_t{ 0 } ),
        m_max_element_id( tjob_t::particle_index_t{ 0 } ),
        m_min_initial_turn_id( tjob_t::particle_index_t{ 0 } ),
        m_max_initial_turn_id( tjob_t::particle_index_t{ 0 } ),
        m_until_turn_elem_by_elem( st_size_t{ 0 } ),
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

    tjob_t::TrackJobBaseNew( TrackJobBaseNew const& other ) :
        st::ArchDebugBase( other ),
        m_assign_address_items( other.m_assign_address_items ),
        m_particle_set_indices( other.m_particle_set_indices ),
        m_particle_set_begin_offsets( other.m_particle_set_begin_offsets ),
        m_particle_set_end_offsets( other.m_particle_set_end_offsets ),
        m_beam_monitor_indices( other.m_beam_monitor_indices ),
        m_stored_buffers( other.m_stored_buffers ),
        m_assign_item_keys( other.m_assign_item_keys ),
        m_my_output_buffer( nullptr ),
        m_my_elem_by_elem_buffer( nullptr ),
        m_my_particles_addr_buffer( nullptr ),
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
        m_elem_by_elem_config_index( other.m_elem_by_elem_config_index ),
        m_num_stored_buffers( other.m_num_stored_buffers ),
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
        if( other.ownsOutputBuffer() )
        {
            this->m_my_output_buffer.reset( new
                tjob_t::buffer_t( *other.ptrOutputBuffer() ) );

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

        tjob_t::COPY_PTR_BUFFER( this->m_my_elem_by_elem_buffer,
                                 other.m_my_elem_by_elem_buffer );

        tjob_t::COPY_PTR_BUFFER( this->m_my_particles_addr_buffer,
                                 other.m_my_particles_addr_buffer );
    }

    tjob_t::TrackJobBaseNew(
        TrackJobBaseNew&& other ) SIXTRL_NOEXCEPT :
        st::ArchDebugBase( std::move( other ) ),
        m_assign_address_items( std::move( other.m_assign_address_items ) ),
        m_particle_set_indices( std::move( other.m_particle_set_indices ) ),
        m_particle_set_begin_offsets(
            std::move( other.m_particle_set_begin_offsets ) ),
        m_particle_set_end_offsets(
            std::move( other.m_particle_set_end_offsets ) ),
        m_beam_monitor_indices( std::move( other.m_beam_monitor_indices ) ),
        m_stored_buffers( std::move( other.m_stored_buffers ) ),
        m_assign_item_keys( std::move( other.m_assign_item_keys ) ),
        m_my_output_buffer( std::move( other.m_my_output_buffer ) ),
        m_my_elem_by_elem_buffer( std::move( other.m_my_elem_by_elem_buffer ) ),
        m_my_particles_addr_buffer(
            std::move( other.m_my_particles_addr_buffer ) ),
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
        m_elem_by_elem_config_index(
            std::move( other.m_elem_by_elem_config_index ) ),
        m_num_stored_buffers( std::move( other.m_num_stored_buffers ) ),
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

    TrackJobBaseNew& tjob_t::operator=( TrackJobBaseNew const& rhs )
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
                    new tjob_t::buffer_t(
                        *rhs.m_my_particles_addr_buffer ) );
            }
            else
            {
                this->m_my_particles_addr_buffer.reset(
                    new tjob_t::buffer_t );
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

            this->m_elem_by_elem_config_index =
                rhs.m_elem_by_elem_config_index;

            this->m_num_stored_buffers = rhs.m_num_stored_buffers;

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

    TrackJobBaseNew& tjob_t::operator=(
        TrackJobBaseNew&& rhs ) SIXTRL_NOEXCEPT
    {
        if( this != &rhs )
        {
            st::ArchDebugBase::operator=( std::move( rhs ) );

            this->m_assign_address_items =
                std::move( rhs.m_assign_address_items );

            this->m_particle_set_indices =
                std::move( rhs.m_particle_set_indices );

            this->m_particle_set_begin_offsets =
                std::move( rhs.m_particle_set_begin_offsets );

            this->m_particle_set_end_offsets =
                std::move( rhs.m_particle_set_end_offsets );

            this->m_beam_monitor_indices =
                std::move( rhs.m_beam_monitor_indices );

            this->m_stored_buffers = std::move( rhs.m_stored_buffers );
            this->m_assign_item_keys = std::move( rhs.m_assign_item_keys );

            this->m_my_output_buffer = std::move( rhs.m_my_output_buffer );

            this->m_my_elem_by_elem_buffer =
                std::move( rhs.m_my_elem_by_elem_buffer );

            this->m_my_particles_addr_buffer =
                std::move( rhs.m_my_particles_addr_buffer );

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

            this->m_elem_by_elem_config_index =
                std::move( rhs.m_elem_by_elem_config_index );

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

    void tjob_t::doClear( tjob_t::clear_flag_t const flags )
    {
        if( tjob_t::IsClearFlagSet(
                flags, st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES ) )
        {
            this->doClearParticlesStructures();
        }

        if( tjob_t::IsClearFlagSet(
                flags, st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES ) )
        {
            this->doClearBeamElementsStructures();
        }

        if( tjob_t::IsClearFlagSet(
                flags,st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES ) )
        {
            this->doClearOutputStructures();
        }

        return;
    }

    st_collect_flag_t tjob_t::doCollect(
        st_collect_flag_t const flags )
    {
        return ( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    st_push_flag_t tjob_t::doPush(
        st_collect_flag_t const flags )
    {
        return ( flags & st::TRACK_JOB_PUSH_ALL );
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::doPrepareParticlesStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using size_t = st_size_t;
        using p_index_t = tjob_t::particle_index_t;

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
                tjob_t::ptr_particles_addr_buffer_t partaddr_buffer_store(
                    new tjob_t::buffer_t );

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

    void tjob_t::doClearParticlesStructures()
    {
        this->doClearParticlesStructuresBaseImpl();
    }

    st_status_t tjob_t::doPrepareBeamElementsStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using size_t        = st_size_t;
        using p_index_t     = tjob_t::particle_index_t;
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

                num_be_monitors = ::NS(BeamMonitor_monitor_indices_from_buffer)(
                    be_mon_indices.data(), be_mon_indices.size(), belems );

                if( num_be_monitors > be_mon_indices.size() )
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                }

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

    void tjob_t::doClearBeamElementsStructures()
    {
        this->doClearBeamElementsStructuresBaseImpl();
    }

    st_status_t tjob_t::doPrepareOutputStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        using size_t                 = st_size_t;
        using buffer_t               = tjob_t::buffer_t;
        using c_buffer_t             = tjob_t::c_buffer_t;
        using buf_size_t             = ::NS(buffer_size_t);
        using elem_by_elem_config_t  = tjob_t::elem_by_elem_config_t;
        using ptr_output_buffer_t    = tjob_t::ptr_output_buffer_t;
        using par_index_t            = tjob_t::particle_index_t;

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

            SIXTRL_ASSERT( this->m_my_elem_by_elem_buffer.get() != nullptr );
            this->m_my_elem_by_elem_buffer->reset();

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( until_turn_elem_by_elem > ZERO ) &&
                ( this->minInitialTurnId() >= par_index_t{ 0 } ) &&
                ( max_elem_by_elem_turn_id >= this->minInitialTurnId() ) &&
                ( until_turn_elem_by_elem > static_cast< buf_size_t >(
                    this->minInitialTurnId() ) ) )
            {
                SIXTRL_ASSERT( this->m_my_elem_by_elem_buffer.get() !=
                    nullptr );

                elem_by_elem_config_t* conf = ::NS(ElemByElemConfig_preset)(
                    ::NS(ElemByElemConfig_new)(
                        this->m_my_elem_by_elem_buffer->getCApiPtr() ) );

                SIXTRL_ASSERT( conf != nullptr );

                status = ::NS(ElemByElemConfig_init_detailed)( conf,
                    this->defaultElemByElemOrder(),
                    this->minParticleId(), this->maxParticleId(),
                    this->minElementId(),  this->maxElementId(),
                    this->minInitialTurnId(), max_elem_by_elem_turn_id,
                    this->defaultElemByElemRolling() );

                this->m_elem_by_elem_config_index = size_t{ 0 };

                if( status == st::ARCH_STATUS_SUCCESS )
                {
                    this->doSetUntilTurnElemByElem( until_turn_elem_by_elem );
                    SIXTRL_ASSERT( this->hasElemByElemConfig() );
                }
            }
            else if( status == st::ARCH_STATUS_SUCCESS )
            {
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

    void tjob_t::doClearOutputStructures()
    {
        this->doClearOutputStructuresBaseImpl();
    }

    st_status_t tjob_t::doAssignOutputBufferToBeamMonitors(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        tjob_t::particle_index_t const min_turn_id,
        st_size_t const output_buffer_offset_index )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        this->doSetBeamMonitorOutputEnabledFlag( false );

        if( ( output_buffer != nullptr ) && ( beam_elem_buffer != nullptr ) &&
            ( this->numBeamMonitors() > st_size_t{ 0 } ) &&
            ( min_turn_id >= tjob_t::particle_index_t{ 0 } ) &&
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


    st_status_t tjob_t::doAssignOutputBufferToElemByElemConfig(
        tjob_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        st_size_t const output_buffer_offset_index )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

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

    tjob_t::clear_flag_t tjob_t::doPrepareResetClearFlags(
        const tjob_t::c_buffer_t *const SIXTRL_RESTRICT pbuffer,
        st_size_t const num_psets,
        st_size_t const* SIXTRL_RESTRICT pset_indices_begin,
        const tjob_t::c_buffer_t *const SIXTRL_RESTRICT belems_buffer,
        const tjob_t::c_buffer_t *const SIXTRL_RESTRICT output_buffer,
        st_size_t const )
    {
        using clear_flag_t = tjob_t::clear_flag_t;
        clear_flag_t clear_flags = this->doGetDefaultPrepareResetClearFlags();

        bool const has_same_particle_sets = (
            ( pset_indices_begin != nullptr ) &&
            ( this->numParticleSets() == num_psets ) &&
            ( ( pset_indices_begin == this->particleSetIndicesBegin() ) ||
              ( std::equal( pset_indices_begin, pset_indices_begin + num_psets,
                            this->particleSetIndicesBegin() ) ) ) );

        if( ( tjob_t::IsClearFlagSet( clear_flags,
                st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES ) ) &&
            ( pbuffer != nullptr ) &&
            ( pbuffer == this->ptrCParticlesBuffer() ) &&
            ( has_same_particle_sets ) )
        {
            clear_flags = tjob_t::UnsetClearFlag( clear_flags,
               st::TRACK_JOB_CLEAR_PARTICLE_STRUCTURES );
        }

        if( ( tjob_t::IsClearFlagSet( clear_flags,
                st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES ) ) &&
            ( belems_buffer != nullptr ) &&
            ( belems_buffer == this->ptrCBeamElementsBuffer() ) )
        {
            clear_flags = tjob_t::UnsetClearFlag( clear_flags,
               st::TRACK_JOB_CLEAR_BEAM_ELEMENT_STRUCTURES );
        }

        if( ( tjob_t::IsClearFlagSet( clear_flags,
                st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES ) ) &&
            ( output_buffer != nullptr ) &&
            ( output_buffer == this->ptrCOutputBuffer() ) )
        {
            clear_flags = tjob_t::UnsetClearFlag( clear_flags,
               st::TRACK_JOB_CLEAR_OUTPUT_STRUCTURES );
        }

        return clear_flags;
    }

    st_status_t tjob_t::doReset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        using output_buffer_flag_t = tjob_t::output_buffer_flag_t;

        st_status_t status =
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

    st_status_t tjob_t::doAssignNewOutputBuffer(
        c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        using flags_t = tjob_t::output_buffer_flag_t;

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

    st_size_t tjob_t::do_add_stored_buffer(
        tjob_t::buffer_store_t&& buffer_store_handle )
    {
        tjob_t::c_buffer_t* ptr_cbuffer_null = nullptr;
        st_size_t buffer_index = st::ARCH_MIN_USER_DEFINED_BUFFER_ID +
            this->m_stored_buffers.size();

        if( buffer_store_handle.active() ) ++this->m_num_stored_buffers;
        this->m_stored_buffers.emplace_back( ptr_cbuffer_null, false );
        this->m_stored_buffers.back() = std::move( buffer_store_handle );
        return buffer_index;
    }

    st_status_t tjob_t::do_remove_stored_buffer( st_size_t const buffer_idx )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        st_size_t const min_buffer_id = this->min_stored_buffer_id();
        st_size_t const max_buffer_id_plus_one =
            min_buffer_id + this->m_stored_buffers.size();

        if( ( min_buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_idx != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_idx >= min_buffer_id ) &&
            ( buffer_idx < max_buffer_id_plus_one ) )
        {
            st_size_t const ii = buffer_idx - min_buffer_id;
            if( this->m_stored_buffers[ ii ].active() )
            {
                SIXTRL_ASSERT( this->m_num_stored_buffers > st_size_t{ 0 } );
                this->m_stored_buffers[ ii ].clear();
                --this->m_num_stored_buffers;
            }

            status = st::ARCH_STATUS_SUCCESS;
        }

        return status;
    }

    st_status_t tjob_t::do_push_stored_buffer( st_size_t const buffer_id )
    {
        tjob_t::buffer_store_t const* ptr_stored_buffer =
            this->do_get_ptr_buffer_store( buffer_id );

        return ( ptr_stored_buffer != nullptr )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::do_collect_stored_buffer( st_size_t const buffer_id )
    {
        tjob_t::buffer_store_t const* ptr_stored_buffer =
            this->do_get_ptr_buffer_store( buffer_id );

        return ( ptr_stored_buffer != nullptr )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::do_add_assign_address_item(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF assign_item,
        st_size_t* SIXTRL_RESTRICT ptr_item_index )
    {
        using buffer_t = tjob_t::buffer_t;
        using key_t    = tjob_t::assign_item_key_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        size_t item_index = std::numeric_limits< size_t >::max();

        if( assign_item.valid() )
        {
            key_t const key = key_t{ assign_item.getDestBufferId(),
                                     assign_item.getSrcBufferId() };

            auto it = this->m_assign_address_items.find( key );
            if( it == this->m_assign_address_items.end() )
            {
                auto cmp_key_fn = []( key_t const& SIXTRL_RESTRICT_REF lhs,
                                      key_t const& SIXTRL_RESTRICT_REF rhs )
                {
                    return ( ( lhs.dest_buffer_id < rhs.dest_buffer_id ) ||
                             ( ( lhs.dest_buffer_id == rhs.dest_buffer_id ) &&
                               ( lhs.src_buffer_id < rhs.src_buffer_id ) ) );
                };

                SIXTRL_ASSERT( std::is_sorted( this->m_assign_item_keys.begin(),
                    this->m_assign_item_keys.end(), cmp_key_fn ) );

                SIXTRL_ASSERT( !std::binary_search(
                    this->m_assign_item_keys.begin(),
                    this->m_assign_item_keys.end(), key, cmp_key_fn ) );

                SIXTRL_ASSERT( this->m_assign_address_items.size() ==
                               this->m_assign_item_keys.size() );

                bool const keys_need_sorting = !(
                    ( this->m_assign_item_keys.empty() ) ||
                    ( cmp_key_fn( this->m_assign_item_keys.back(), key ) ) );

                this->m_assign_item_keys.push_back( key );

                if( keys_need_sorting )
                {
                    std::sort( this->m_assign_item_keys.begin(),
                               this->m_assign_item_keys.end(), cmp_key_fn );
                }

                buffer_t buffer;

                auto ret = this->m_assign_address_items.emplace(
                    std::make_pair( key, std::move( buffer ) ) );

                if( ret.second )
                {
                    buffer_t& buffer = ret.first->second;
                    SIXTRL_ASSERT( buffer.getNumObjects() == size_t{ 0 } );

                    item_index = buffer.getNumObjects();

                    ::NS(AssignAddressItem)* item =
                        ::NS(AssignAddressItem_add_copy)( buffer.getCApiPtr(),
                            assign_item.getCApiPtr() );

                    if( ( item != nullptr ) &&
                        ( buffer.getNumObjects() > item_index ) )
                    {
                        status = st::ARCH_STATUS_SUCCESS;
                    }
                }
            }
            else
            {
                item_index = this->do_find_assign_address_item( assign_item );
                if( item_index < it->second.getNumObjects() )
                {
                    status = st::ARCH_STATUS_SUCCESS;
                }
                else
                {
                    item_index = it->second.getNumObjects();

                    ::NS(AssignAddressItem)* item =
                        ::NS(AssignAddressItem_add_copy)(
                            it->second.getCApiPtr(), assign_item.getCApiPtr() );

                    if( ( item != nullptr ) &&
                        ( item_index < it->second.getNumObjects() ) )
                    {
                        status = st::ARCH_STATUS_SUCCESS;
                    }
                }
            }

            if( ( status == st::ARCH_STATUS_SUCCESS ) &&
                ( ptr_item_index != nullptr ) )
            {
                *ptr_item_index = item_index;
            }
        }

        return status;
    }

    st_status_t tjob_t::do_remove_assign_address_item(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key,
        st_size_t const index_of_item_to_remove )
    {
        using item_t = tjob_t::assign_item_t;
        using key_t  = tjob_t::assign_item_key_t;

        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        auto it = this->m_assign_address_items.find( key );

        auto cmp_key_fn = []( key_t const& SIXTRL_RESTRICT_REF lhs,
                              key_t const& SIXTRL_RESTRICT_REF rhs )
        {
            return ( ( lhs.dest_buffer_id < rhs.dest_buffer_id ) ||
                     ( ( lhs.dest_buffer_id == rhs.dest_buffer_id ) &&
                       ( lhs.src_buffer_id < rhs.src_buffer_id ) ) );
        };

        SIXTRL_ASSERT( std::is_sorted( this->m_assign_item_keys.begin(),
                this->m_assign_item_keys.end(), cmp_key_fn ) );

        if( ( it != this->m_assign_address_items.end() ) &&
            ( it->second.getNumObjects() > index_of_item_to_remove ) )
        {
            buffer_t& current_buffer = it->second;
            buffer_t new_buffer( current_buffer.getNumObjects(),
                                 current_buffer.getNumSlots(),
                                 current_buffer.getNumDataptrs(),
                                 current_buffer.getNumGarbageRanges(),
                                 current_buffer.getFlags() );

            size_t const nn = current_buffer.getNumObjects();
            status = st::ARCH_STATUS_SUCCESS;

            SIXTRL_ASSERT( std::binary_search( this->m_assign_item_keys.begin(),
                this->m_assign_item_keys.end(), key, cmp_key_fn ) );

            SIXTRL_ASSERT( this->m_assign_address_items.size() ==
                           this->m_assign_item_keys.size() );

            for( size_t ii = size_t{ 0 } ; ii < nn ; ++ii )
            {
                if( ii == index_of_item_to_remove ) continue;
                item_t const* in_item = current_buffer.get< item_t >( ii );

                if( in_item == nullptr )
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                    break;
                }

                ::NS(AssignAddressItem)* copied_item =
                    ::NS(AssignAddressItem_add_copy)(
                        current_buffer.getCApiPtr(), in_item->getCApiPtr() );

                if( copied_item == nullptr )
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                    break;
                }
            }

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                using std::swap;
                swap( it->second, new_buffer );
            }
        }

        if( status == st::ARCH_STATUS_SUCCESS )
        {
            it = this->m_assign_address_items.find( key );

            if( it->second.getNumObjects() == size_t{ 0 } )
            {
                this->m_assign_address_items.erase( key );

                auto key_it = std::lower_bound(
                    this->m_assign_item_keys.begin(),
                    this->m_assign_item_keys.end(), key, cmp_key_fn );

                SIXTRL_ASSERT( key_it != this->m_assign_item_keys.end() );
                this->m_assign_item_keys.erase( key_it );
            }
        }

        return status;
    }

    st_status_t tjob_t::do_perform_address_assignments(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key )
    {
        using c_buffer_t = tjob_t::c_buffer_t;
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        size_t const dest_buffer_id = key.dest_buffer_id;
        c_buffer_t* dest_buffer = this->buffer_by_buffer_id( dest_buffer_id );

        size_t const src_buffer_id = key.src_buffer_id;
        c_buffer_t const* src_buffer =
            this->buffer_by_buffer_id( src_buffer_id );

        auto it = this->m_assign_address_items.find( key );

        if( ( it != this->m_assign_address_items.end() ) &&
            ( dest_buffer != nullptr ) && ( src_buffer != nullptr ) )
        {
            unsigned char* dest_buffer_begin =
            ::NS(Buffer_get_data_begin)( dest_buffer );

            unsigned char const* src_buffer_begin =
                ::NS(Buffer_get_const_data_begin)( src_buffer );

            size_t const dest_slot_size =
                ( this->is_buffer_by_buffer_id( dest_buffer_id ) )
                    ? ::NS(Buffer_get_slot_size)( dest_buffer ) : size_t{ 0 };

            size_t const src_slot_size =
                ( this->is_buffer_by_buffer_id( src_buffer_id ) )
                    ? ::NS(Buffer_get_slot_size)( src_buffer ) : size_t{ 0 };

            SIXTRL_ASSERT( dest_buffer_begin != nullptr );
            SIXTRL_ASSERT( src_buffer_begin  != nullptr );

            status =
            ::NS(AssignAddressItem_perform_address_assignment_kernel_impl)(
                it->second.dataBegin< unsigned char const* >(),
                it->second.getSlotSize(), st_size_t{ 0 }, st_size_t{ 1 },
                dest_buffer_begin, dest_slot_size, dest_buffer_id,
                src_buffer_begin, src_slot_size, src_buffer_id );
        }

        return status;
    }

    st_status_t tjob_t::do_commit_address_assignments()
    {
        return st::ARCH_STATUS_SUCCESS;
    }

    st_status_t tjob_t::do_rebuild_assign_items_buffer_arg()
    {
        return st::ARCH_STATUS_SUCCESS;
    }

    /* --------------------------------------------------------------------- */

    st_status_t tjob_t::doFetchParticleAddresses()
    {
        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::doClearParticleAddresses(
        st_size_t const index )
    {
        using status_t = st_status_t;

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

    st_status_t tjob_t::doClearAllParticleAddresses()
    {
        using status_t = st_status_t;

        status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->doGetPtrParticlesAddrBuffer() != nullptr )
        {
            status = ::NS(ParticlesAddr_buffer_clear_all)(
                this->doGetPtrParticlesAddrBuffer()->getCApiPtr() );

            this->doSetHasParticleAddressesFlag( false );
        }

        return status;
    }

    tjob_t::track_status_t tjob_t::doTrackUntilTurn(
        st_size_t const )
    {
        return st::TRACK_STATUS_GENERAL_FAILURE;
    }

    tjob_t::track_status_t tjob_t::doTrackElemByElem(
        st_size_t const until_turn_elem_by_elem )
    {
        ( void )until_turn_elem_by_elem;
        return st::TRACK_STATUS_GENERAL_FAILURE;
    }

    tjob_t::track_status_t tjob_t::doTrackLine(
        st_size_t const, st_size_t const,
            bool const )
    {
        return st::TRACK_STATUS_GENERAL_FAILURE;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doSetPtrParticlesBuffer(
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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

    void tjob_t::doSetPtrBeamElementsBuffer(
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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

    void tjob_t::doSetPtrOutputBuffer(
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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

    void tjob_t::doSetCxxBufferPointers(
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer ) SIXTRL_NOEXCEPT
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

    void tjob_t::doSetPtrCParticlesBuffer( tjob_t::c_buffer_t*
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

    void tjob_t::doSetPtrCBeamElementsBuffer(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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

    void tjob_t::doSetPtrCOutputBuffer( tjob_t::c_buffer_t*
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

    void tjob_t::doSetBeamMonitorOutputBufferOffset(
        st_size_t const
            output_buffer_offset ) SIXTRL_NOEXCEPT
    {
        this->m_be_mon_output_buffer_offset = output_buffer_offset;
    }

    void tjob_t::doSetElemByElemOutputIndexOffset(
        st_size_t const
            elem_by_elem_output_offset ) SIXTRL_NOEXCEPT
    {
        this->m_elem_by_elem_output_offset = elem_by_elem_output_offset;
    }

    void tjob_t::doSetUntilTurnElemByElem(
        tjob_t::particle_index_t
            const until_turn_elem_by_elem ) SIXTRL_NOEXCEPT
    {
        this->m_until_turn_elem_by_elem = until_turn_elem_by_elem;
    }

    void tjob_t::doSetRequiresCollectFlag(
        bool const requires_collect_flag ) SIXTRL_NOEXCEPT
    {
        this->m_requires_collect = requires_collect_flag;
    }

    void tjob_t::doSetBeamMonitorOutputEnabledFlag(
        bool const has_beam_monitor_output ) SIXTRL_NOEXCEPT
    {
        this->m_has_beam_monitor_output = has_beam_monitor_output;
    }

    void tjob_t::doSetElemByElemOutputEnabledFlag(
        bool const has_elem_by_elem_output ) SIXTRL_NOEXCEPT
    {
        this->m_has_elem_by_elem_output = has_elem_by_elem_output;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doInitDefaultParticleSetIndices()
    {
        this->m_particle_set_indices.clear();
        this->m_particle_set_indices.push_back( st_size_t{ 0 } );

        this->m_particle_set_begin_offsets.clear();
        this->m_particle_set_begin_offsets.push_back(
            tjob_t::num_particles_t{ 0 } );

        this->m_particle_set_end_offsets.clear();
        this->m_particle_set_end_offsets.push_back(
            tjob_t::num_particles_t{ 0 } );
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doInitDefaultBeamMonitorIndices()
    {
        this->m_beam_monitor_indices.clear();
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doSetNumParticleSetsInBuffer(
        st_size_t const num_psets ) SIXTRL_NOEXCEPT
    {
        this->m_num_particle_sets_in_buffer = num_psets;
    }

    SIXTRL_HOST_FN void tjob_t::doSetNumBeamElementsInBuffer(
        st_size_t const num_belems ) SIXTRL_NOEXCEPT
    {
        this->m_num_beam_elements_in_buffer = num_belems;
    }

    void tjob_t::doSetTotalNumParticles(
        tjob_t::num_particles_t const num_particles ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles = num_particles;
    }

    void tjob_t::doSetTotalNumParticlesInSets(
        tjob_t::num_particles_t const pnum_in_sets ) SIXTRL_NOEXCEPT
    {
        this->m_total_num_particles_in_sets = pnum_in_sets;
    }

    void tjob_t::doSetMinParticleId(
        tjob_t::particle_index_t const min_part_id ) SIXTRL_NOEXCEPT
    {
         this->m_min_particle_id = min_part_id;
    }

    void tjob_t::doSetMaxParticleId(
        tjob_t::particle_index_t const max_part_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_particle_id = max_part_id;
    }

    void tjob_t::doSetMinElementId(
        tjob_t::particle_index_t const min_elem_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_element_id = min_elem_id;
    }

    void tjob_t::doSetMaxElementId(
        tjob_t::particle_index_t const max_elem_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_element_id = max_elem_id;
    }

    void tjob_t::doSetMinInitialTurnId(
        tjob_t::particle_index_t const
            min_initial_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_initial_turn_id = min_initial_turn_id;
    }

    void tjob_t::doSetMaxInitialTurnId( particle_index_t const
        max_initial_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_initial_turn_id = max_initial_turn_id;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::buffer_t const*
        tjob_t::doGetPtrParticlesAddrBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_my_particles_addr_buffer.get();
    }

    tjob_t::buffer_t*
    tjob_t::doGetPtrParticlesAddrBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_my_particles_addr_buffer.get();
    }

    void tjob_t::doUpdateStoredParticlesAddrBuffer(
        tjob_t::ptr_particles_addr_buffer_t&&
            ptr_buffer ) SIXTRL_NOEXCEPT
    {
        this->m_my_particles_addr_buffer = std::move( ptr_buffer );
    }

    void tjob_t::doSetHasParticleAddressesFlag(
            bool const has_particle_addresses ) SIXTRL_NOEXCEPT
    {
        this->m_has_particle_addresses = has_particle_addresses;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doUpdateStoredOutputBuffer(
        tjob_t::ptr_output_buffer_t&& ptr_out_buffer ) SIXTRL_NOEXCEPT
    {
        this->doSetPtrOutputBuffer( ptr_out_buffer.get() );
        this->m_my_output_buffer = std::move( ptr_out_buffer );
    }

    void tjob_t::doSetUsesControllerFlag(
        bool const uses_controller_flag ) SIXTRL_NOEXCEPT
    {
        this->m_uses_controller = uses_controller_flag;
    }

    void tjob_t::doSetUsesArgumentsFlag(
        bool const arguments_flag ) SIXTRL_NOEXCEPT
    {
        this->m_uses_arguments = arguments_flag;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doClearParticlesStructuresBaseImpl() SIXTRL_NOEXCEPT
    {
        using num_particles_t = tjob_t::num_particles_t;
        using size_t = st_size_t;

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

    void tjob_t::doClearBeamElementsStructuresBaseImpl() SIXTRL_NOEXCEPT
    {
        using size_t = st_size_t;

        this->doSetPtrBeamElementsBuffer( nullptr );
        this->doSetPtrCBeamElementsBuffer( nullptr );

        this->doInitDefaultBeamMonitorIndices();
        this->doSetNumBeamElementsInBuffer( size_t{ 0 } );
    }

    void tjob_t::doClearOutputStructuresBaseImpl() SIXTRL_NOEXCEPT
    {
        using size_t = st_size_t;

        this->doSetPtrOutputBuffer( nullptr );
        this->doSetPtrCOutputBuffer( nullptr );

        this->doSetBeamMonitorOutputEnabledFlag( false );
        this->doSetElemByElemOutputEnabledFlag( false );

        this->doSetBeamMonitorOutputBufferOffset( size_t{ 0 } );
        this->doSetElemByElemOutputIndexOffset( size_t{ 0 } );
        this->doSetUntilTurnElemByElem( size_t{ 0 } );

        SIXTRL_ASSERT( this->m_my_elem_by_elem_buffer.get() != nullptr );
        this->m_my_elem_by_elem_buffer->reset();
        this->m_elem_by_elem_config_index = size_t{ 0 };

        this->m_my_output_buffer.reset( nullptr );

        this->m_default_elem_by_elem_order   =
            ::NS(ELEM_BY_ELEM_ORDER_DEFAULT);

        this->m_default_elem_by_elem_rolling = true;

        return;
    }

    /* ********************************************************************* */

    TrackJobBaseNew* TrackJobNew_create( tjob_t::arch_id_t const arch_id,
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

        ( void )conf_str;
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

    TrackJobBaseNew* TrackJobNew_new( ::NS(arch_id_t) const arch_id,
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
        using size_t = st_size_t;

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
