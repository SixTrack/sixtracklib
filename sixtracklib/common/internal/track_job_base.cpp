#include "sixtracklib/common/internal/track_job_base.h"

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <iostream>
        #include <memory>
        #include <numeric>
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
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/track/definitions.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
    #include "sixtracklib/common/buffer/assign_address_item_kernel_impl.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/be_monitor/output_buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/particles/particles_addr.h"
    #include "sixtracklib/common/internal/stl_buffer_helper.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        namespace st      = SIXTRL_CXX_NAMESPACE;
        using tjob_t      = st::TrackJobBase;
        using st_size_t   = st_size_t;
        using st_status_t = tjob_t::status_t;
    }

    constexpr st_size_t tjob_t::ILLEGAL_BUFFER_ID;

    void tjob_t::COPY_PTR_BUFFER(
        tjob_t::ptr_buffer_t& SIXTRL_RESTRICT_REF dest_ptr_buffer,
        tjob_t::ptr_buffer_t const& SIXTRL_RESTRICT_REF src_ptr_buffer )
    {
        if( src_ptr_buffer.get() != nullptr )
        {
            dest_ptr_buffer.reset(
                new tjob_t::buffer_t( *src_ptr_buffer.get() ) );
        }
        else
        {
            dest_ptr_buffer.reset( nullptr );
        }
    }

    /* --------------------------------------------------------------------- */

    st_size_t tjob_t::DefaultNumParticleSetIndices() SIXTRL_NOEXCEPT
    {
        return st::TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS;
    }

    st_size_t const*
    tjob_t::DefaultParticleSetIndicesBegin() SIXTRL_NOEXCEPT
    {
        st_size_t const* ptr = &st::TRACK_JOB_DEFAULT_PARTICLE_SET_INDICES[ 0 ];
        return ptr;
    }

    st_size_t const* tjob_t::DefaultParticleSetIndicesEnd() SIXTRL_NOEXCEPT
    {
        st_size_t const* end_ptr = tjob_t::DefaultParticleSetIndicesBegin();
        std::advance( end_ptr, st::TRACK_JOB_DEFAULT_NUM_PARTICLE_SETS );
        return end_ptr;
    }

    void tjob_t::clear()
    {
        this->doClear();
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::collect()
    {
        this->doCollect( this->m_collect_flags );
    }

    void tjob_t::collect( tjob_t::collect_flag_t const flags )
    {
        this->doCollect( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    void tjob_t::collectParticles()
    {
        this->doCollect( st::TRACK_JOB_IO_PARTICLES );
    }


    void tjob_t::collectBeamElements()
    {
        this->doCollect( st::TRACK_JOB_IO_BEAM_ELEMENTS );
    }

    void tjob_t::collectOutput()
    {
        this->doCollect( st::TRACK_JOB_IO_OUTPUT );
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

    tjob_t::collect_flag_t tjob_t::collectFlags() const SIXTRL_NOEXCEPT
    {
        return this->m_collect_flags;
    }

    void tjob_t::setCollectFlags(
        tjob_t::collect_flag_t const flags ) SIXTRL_NOEXCEPT
    {
        this->m_collect_flags = ( flags & st::TRACK_JOB_COLLECT_ALL );
    }

    bool tjob_t::requiresCollecting() const SIXTRL_NOEXCEPT
    {
        return this->m_requires_collect;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::push( tjob_t::push_flag_t const flags )
    {
        this->doPush( flags );
    }

    void tjob_t::pushParticles()
    {
        this->doPush( st::TRACK_JOB_IO_PARTICLES );
    }

    void tjob_t::pushBeamElements()
    {
        this->doPush( st::TRACK_JOB_IO_BEAM_ELEMENTS );
    }

    void tjob_t::pushOutput()
    {
        this->doPush( st::TRACK_JOB_IO_OUTPUT );
    }

    /* --------------------------------------------------------------------- */

    tjob_t::track_status_t tjob_t::track( st_size_t const until_turn )
    {
        return this->doTrackUntilTurn( until_turn );
    }

    tjob_t::track_status_t tjob_t::trackElemByElem(
        st_size_t const until_turn )
    {
        return this->doTrackElemByElem( until_turn );
    }

    tjob_t::track_status_t tjob_t::trackLine(
        st_size_t const beam_elements_begin_index,
        st_size_t const beam_elements_end_index, bool const finish_turn )
    {
        return this->doTrackLine(
            beam_elements_begin_index, beam_elements_end_index, finish_turn );
    }

    bool tjob_t::reset(
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF be_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        st_size_t const particle_set_indices[] = { st_size_t{ 0 }, st_size_t{ 0 } };

        return this->reset( particles_buffer,
            &particle_set_indices[ 0 ], &particle_set_indices[ 1 ], be_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
    }

    bool tjob_t::reset(
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        st_size_t const pset_index,
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF be_buffer,
        tjob_t::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        st_size_t const particle_set_indices[] = { pset_index, st_size_t{ 0 } };

        return this->reset( particles_buffer, &particle_set_indices[ 0 ],
            &particle_set_indices[ 1 ], be_buffer, ptr_output_buffer,
                until_turn_elem_by_elem );
    }

    bool tjob_t::reset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        st_size_t const pset_index = st_size_t{ 0 };
        return this->reset( particles_buffer, st_size_t{ 1 }, &pset_index,
            be_buffer, ptr_output_buffer, until_turn_elem_by_elem );
    }

    bool tjob_t::reset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        st_size_t const particle_set_index,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        return this->reset( particles_buffer,
            st_size_t{ 1 }, &particle_set_index, be_buffer,
                ptr_output_buffer, until_turn_elem_by_elem );
    }

    bool tjob_t::reset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        st_size_t const num_particle_sets,
        st_size_t const* SIXTRL_RESTRICT
            particle_set_indices_begin,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        st_size_t const* particle_set_indices_end =
            particle_set_indices_begin;

        if( ( particle_set_indices_end != nullptr ) &&
            ( num_particle_sets > st_size_t{ 0 } ) )
        {
            std::advance( particle_set_indices_end, num_particle_sets );
        }

        return this->reset( particles_buffer,
            particle_set_indices_begin, particle_set_indices_end,
            be_buffer, ptr_output_buffer, until_turn_elem_by_elem );
    }

    bool tjob_t::selectParticleSet( st_size_t const particle_set_index )
    {
        using buffer_t   = tjob_t::buffer_t;
        using c_buffer_t = tjob_t::c_buffer_t;
        using size_t = st_size_t;

        bool success = false;

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

                success = this->reset( *ptr_particles_buffer,
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

                success = this->reset( ptr_c_particles_buffer, size_t{ 1 },
                    &particle_set_index, ptr_c_beam_elem_buffer,
                        ptr_c_output_buffer, this->untilTurnElemByElem() );
            }
        }

        return success;
    }

    bool tjob_t::assignOutputBuffer(
        tjob_t::buffer_t& SIXTRL_RESTRICT_REF output_buffer )
    {
        bool success = false;

        if( this->doAssignNewOutputBuffer( output_buffer.getCApiPtr() ) )
        {
            if( this->hasOutputBuffer() )
            {
                SIXTRL_ASSERT( output_buffer.getCApiPtr() ==
                    this->ptrCOutputBuffer() );

                this->doSetPtrOutputBuffer( &output_buffer );
            }

            success = true;
        }

        return success;
    }

    bool tjob_t::assignOutputBuffer(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        return this->doAssignNewOutputBuffer( ptr_output_buffer );
    }

    /* --------------------------------------------------------------------- */

    tjob_t::type_t tjob_t::type() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id;
    }

    std::string const& tjob_t::typeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_str;
    }

    char const* tjob_t::ptrTypeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_str.c_str();
    }

    bool tjob_t::hasDeviceIdStr() const SIXTRL_NOEXCEPT
    {
        return ( !this->m_device_id_str.empty() );
    }

    std::string const& tjob_t::deviceIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_device_id_str;
    }

    char const* tjob_t::ptrDeviceIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_device_id_str.c_str();
    }

    bool tjob_t::hasConfigStr() const SIXTRL_NOEXCEPT
    {
        return ( !this->m_config_str.empty() );
    }

    std::string const& tjob_t::configStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str;
    }

    char const* tjob_t::ptrConfigStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str.c_str();
    }

    /* --------------------------------------------------------------------- */

    st_size_t tjob_t::numParticleSets() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_num_particles_in_sets.size() ==
                       this->m_particle_set_indices.size() );

        return this->m_particle_set_indices.size();
    }

    st_size_t const* tjob_t::particleSetIndicesBegin() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_num_particles_in_sets.size() ==
                       this->m_particle_set_indices.size() );

        return this->m_particle_set_indices.data();
    }

    st_size_t const* tjob_t::particleSetIndicesEnd() const SIXTRL_NOEXCEPT
    {
        st_size_t const* end_ptr = this->particleSetIndicesBegin();
        SIXTRL_ASSERT( end_ptr != nullptr );
        std::advance( end_ptr, this->numParticleSets() );
        return end_ptr;
    }

    st_size_t tjob_t::particleSetIndex(
        st_size_t const n ) const
    {
        SIXTRL_ASSERT( this->m_num_particles_in_sets.size() ==
                       this->m_particle_set_indices.size() );

        return this->m_particle_set_indices.at( n );
    }


    st_size_t const* tjob_t::numParticlesInSetsBegin() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_num_particles_in_sets.size() ==
                       this->m_particle_set_indices.size() );

        return this->m_num_particles_in_sets.data();
    }

    st_size_t const* tjob_t::numParticlesInSetsEnd() const SIXTRL_NOEXCEPT
    {
        st_size_t const* end_ptr = this->numParticlesInSetsBegin();
        SIXTRL_ASSERT( end_ptr != nullptr );
        std::advance( end_ptr, this->numParticleSets() );
        return end_ptr;
    }

    st_size_t tjob_t::numParticlesInSet( st_size_t const n ) const
    {
        SIXTRL_ASSERT( this->m_num_particles_in_sets.size() ==
                       this->m_particle_set_indices.size() );

        return this->m_num_particles_in_sets.at( n );
    }

    st_size_t tjob_t::totalNumParticlesInSets() const
    {
        SIXTRL_ASSERT( this->m_num_particles_in_sets.size() ==
                       this->m_particle_set_indices.size() );

        return this->m_total_num_particles_in_sets;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::particle_index_t tjob_t::minParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_particle_id;
    }

    tjob_t::particle_index_t tjob_t::maxParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_particle_id;
    }

    tjob_t::particle_index_t tjob_t::minElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_min_element_id;
    }

    tjob_t::particle_index_t tjob_t::maxElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_max_element_id;
    }

    tjob_t::particle_index_t tjob_t::minInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_initial_turn_id;
    }

    tjob_t::particle_index_t tjob_t::maxInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_initial_turn_id;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::buffer_t* tjob_t::ptrParticlesBuffer() SIXTRL_NOEXCEPT
    {
        using ptr_t   = tjob_t::buffer_t*;

        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).ptrParticlesBuffer() );
    }

    tjob_t::buffer_t const* tjob_t::ptrParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_particles_buffer;
    }

    tjob_t::c_buffer_t* tjob_t::ptrCParticlesBuffer() SIXTRL_NOEXCEPT
    {
        using ptr_t   = tjob_t::c_buffer_t*;

        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).ptrCParticlesBuffer() );
    }

    tjob_t::c_buffer_t const* tjob_t::ptrCParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_c_particles_buffer;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::buffer_t* tjob_t::ptrBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        using ptr_t   = tjob_t::buffer_t*;

        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).ptrBeamElementsBuffer() );
    }

    tjob_t::buffer_t const* tjob_t::ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_beam_elem_buffer;
    }

    tjob_t::c_buffer_t* tjob_t::ptrCBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        using ptr_t   = tjob_t::c_buffer_t*;

        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).ptrCBeamElementsBuffer() );
    }

    tjob_t::c_buffer_t const*
    tjob_t::ptrCBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_c_beam_elem_buffer;
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::can_fetch_particles_addr() const SIXTRL_NOEXCEPT
    {
        return false;
    }

    bool tjob_t::has_particles_addr() const SIXTRL_NOEXCEPT
    {
        return false;
    }

    st_status_t tjob_t::fetch_particles_addr()
    {
        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::clear_all_particles_addr()
    {
        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::clear_particles_addr( st_size_t const )
    {
        return st::ARCH_STATUS_GENERAL_FAILURE;
    }

    tjob_t::particles_addr_t const* tjob_t::particles_addr(
        st_size_t const ) const SIXTRL_NOEXCEPT
    {
        return nullptr;
    }

    tjob_t::buffer_t const*
    tjob_t::ptr_particles_addr_buffer() const SIXTRL_NOEXCEPT
    {
        return nullptr;
    }

    tjob_t::c_buffer_t const*
    tjob_t::ptr_particles_addr_cbuffer() const SIXTRL_NOEXCEPT
    {
        return nullptr;
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

            st_size_t item_index =
                std::numeric_limits< st_size_t >::max();

            st_status_t const status = this->doAddAssignAddressItem(
                assign_item_to_add, &item_index );

            if( status == st::ARCH_STATUS_SUCCESS )
            {
                item = this->doGetAssignAddressItem( key, item_index );
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
        tjob_t::assign_item_key_t const key =
        tjob_t::assign_item_key_t{
            item_to_remove.getDestBufferId(),
            item_to_remove.getSrcBufferId() };

        st_size_t const item_index =
            this->doFindAssingAddressItem( item_to_remove );

        return this->doRemoveAssignAddressItem( key, item_index );
    }

    st_status_t tjob_t::remove_assign_address_item(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key,
        st_size_t const index_of_item_to_remove )
    {
        return this->doRemoveAssignAddressItem( key, index_of_item_to_remove );
    }

    bool tjob_t::has_assign_address_item( tjob_t::assign_item_t const&
        SIXTRL_RESTRICT_REF assign_item ) const SIXTRL_NOEXCEPT
    {
        st_size_t const item_index =
            this->doFindAssingAddressItem( assign_item );

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

    st_size_t tjob_t::index_of_assign_address_item(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF
            assign_item ) const SIXTRL_NOEXCEPT
    {
        return this->doFindAssingAddressItem( assign_item );
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
        return this->doFindAssingAddressItem( tjob_t::assign_item_t{
            dest_type_id, dest_buffer_id, dest_elem_index, dest_pointer_offset,
            src_type_id, src_buffer_id, src_elem_index, src_pointer_offset } );
    }

    bool tjob_t::has_assign_items( st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id ) const SIXTRL_NOEXCEPT
    {
        return ( this->num_assign_items(
            dest_buffer_id, src_buffer_id ) > st_size_t{ 0 } );
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

    st_size_t
    tjob_t::num_distinct_available_assign_address_items_dest_src_pairs()
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
                ptr_buffer = this->m_elem_by_elem_buffer.get();
                break;
            }

            case st::ARCH_PARTICLE_ADDR_BUFFER_ID:
            {
                ptr_buffer = this->m_particles_addr_buffer.get();
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

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object) const* tjob_t::assign_items_begin(
        st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id ) const SIXTRL_NOEXCEPT
    {
        auto it = this->m_assign_address_items.find(
            tjob_t::assign_item_key_t{ dest_buffer_id, src_buffer_id } );

        return ( it != this->m_assign_address_items.end() )
            ? ::NS(Buffer_get_const_objects_begin)( it->second.getCApiPtr() )
            : nullptr;
    }

    SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object) const* tjob_t::assign_items_end(
        st_size_t const dest_buffer_id,
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
            this->doFindAssingAddressItem( assign_address_item );

        return this->doGetAssignAddressItem( tjob_t::assign_item_key_t{
                assign_address_item.getDestBufferId(),
                assign_address_item.getSrcBufferId() }, assign_item_idx );
    }


    tjob_t::assign_item_t const* tjob_t::ptr_assign_address_item(
        st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id,
        st_size_t const assign_item_index ) const SIXTRL_NOEXCEPT
    {
        return this->doGetAssignAddressItem(
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
        return this->doCommitAddressAssignments();
    }

    st_status_t tjob_t::assign_all_addresses()
    {
        st_status_t status = st::ARCH_STATUS_SUCCESS;

        auto it = this->m_assign_address_items.begin();
        auto end = this->m_assign_address_items.end();

        for( ; it != end ; ++it )
        {
            status = this->doPerformAddressAssignments( it->first );
            if( status != st::ARCH_STATUS_SUCCESS ) break;
        }

        return status;
    }

    st_status_t tjob_t::assign_addresses( st_size_t const dest_buffer_id,
        st_size_t const src_buffer_id )
    {
        return this->doPerformAddressAssignments(
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

    bool tjob_t::owns_stored_buffer(
        st_size_t const buffer_id ) const SIXTRL_NOEXCEPT
    {
        auto ptr_buffer_store = this->doGetPtrBufferStore( buffer_id );
        return ( ( ptr_buffer_store != nullptr ) &&
                 ( ptr_buffer_store->owns_buffer() ) );
    }

    st_status_t tjob_t::remove_stored_buffer( st_size_t const buffer_index )
    {
        return this->doRemoveStoredBuffer( buffer_index );
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
            this->doGetPtrBufferStore( buffer_id );

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
            this->doGetPtrBufferStore( buffer_id );

        return ( ptr_stored_buffer != nullptr )
            ? ptr_stored_buffer->ptr_buffer() : nullptr;
    }

    st_status_t tjob_t::push_stored_buffer( st_size_t const buffer_id )
    {
        return this->doPushStoredBuffer( buffer_id );
    }

    st_status_t tjob_t::collect_stored_buffer( st_size_t const buffer_id )
    {
        return this->doCollectStoredBuffer( buffer_id );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasOutputBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCOutputBuffer() != nullptr );
    }

    bool tjob_t::ownsOutputBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_my_output_buffer.get() == nullptr ) ||
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
        SIXTRL_ASSERT(
            (  !this->m_has_beam_monitor_output ) ||
            ( ( this->m_has_beam_monitor_output ) &&
              ( this->m_ptr_c_output_buffer != nullptr ) ) );

        return this->m_has_beam_monitor_output;
    }

    st_size_t tjob_t::beamMonitorsOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_be_mon_output_buffer_offset ) ) ||
            ( this->m_be_mon_output_buffer_offset == st_size_t{ 0 } ) );

        return this->m_be_mon_output_buffer_offset;
    }

    st_size_t tjob_t::elemByElemOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_elem_by_elem_output_offset ) ) ||
            ( this->m_elem_by_elem_output_offset == st_size_t{ 0 } ) );

        return this->m_elem_by_elem_output_offset;
    }

    tjob_t::particle_index_t tjob_t::untilTurnElemByElem() const SIXTRL_NOEXCEPT
    {
        return this->m_until_turn_elem_by_elem;
    }

    st_size_t tjob_t::numElemByElemTurns() const SIXTRL_NOEXCEPT
    {
        using index_t = tjob_t::particle_index_t;

        if( ( this->m_until_turn_elem_by_elem > this->m_min_initial_turn_id ) &&
            ( this->m_min_initial_turn_id >= index_t{ 0 } ) )
        {
            return static_cast< st_size_t >( this->m_until_turn_elem_by_elem -
                this->m_min_initial_turn_id );
        }

        return st_size_t{ 0 };
    }

    tjob_t::buffer_t* tjob_t::ptrOutputBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::buffer_t* >( static_cast< tjob_t const& >(
            *this ).ptrOutputBuffer() );
    }

    tjob_t::buffer_t* tjob_t::ptrOutputBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_output_buffer == nullptr ) ||
            ( this->m_ptr_output_buffer->getCApiPtr() ==
              this->m_ptr_c_output_buffer ) );

        return this->m_ptr_output_buffer;
    }

    tjob_t::c_buffer_t* tjob_t::ptrCOutputBuffer() SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::c_buffer_t* >( static_cast< tjob_t const& >(
            *this ).ptrCOutputBuffer() );
    }

    tjob_t::c_buffer_t const* tjob_t::ptrCOutputBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_output_buffer == nullptr ) ||
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


    st_size_t const* tjob_t::beamMonitorIndicesBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_indices.data();
    }

    st_size_t const* tjob_t::beamMonitorIndicesEnd() const SIXTRL_NOEXCEPT
    {
        st_size_t const* end_ptr = this->beamMonitorIndicesBegin();
        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numBeamMonitors() );
        }

        return end_ptr;
    }

    st_size_t tjob_t::beamMonitorIndex( st_size_t const n ) const
    {
        return this->m_beam_monitor_indices.at( n );
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::hasElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        tjob_t::elem_by_elem_config_t const* conf = this->ptrElemByElemConfig();
        return ( ( conf != nullptr ) &&
                 ( ::NS(ElemByElemConfig_is_active)( conf ) ) );
    }

    tjob_t::c_buffer_t const*
    tjob_t::ptrElemByElemConfigCBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->m_elem_by_elem_buffer.get() != nullptr )
            ? this->m_elem_by_elem_buffer->getCApiPtr() : nullptr;
    }

    tjob_t::c_buffer_t*
    tjob_t::ptrElemByElemConfigCBuffer() SIXTRL_NOEXCEPT
    {
        return ( this->m_elem_by_elem_buffer.get() != nullptr )
            ? this->m_elem_by_elem_buffer->getCApiPtr() : nullptr;
    }

    tjob_t::buffer_t const*
    tjob_t::ptrElemByElemConfigBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_elem_by_elem_buffer.get();
    }

    tjob_t::buffer_t* tjob_t::ptrElemByElemConfigBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_elem_by_elem_buffer.get();
    }

    tjob_t::elem_by_elem_config_t*
    tjob_t::ptrElemByElemConfig() SIXTRL_NOEXCEPT
    {
        using  ptr_t = tjob_t::elem_by_elem_config_t*;
        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).ptrElemByElemConfig() );
    }

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

    bool tjob_t::debugMode() const SIXTRL_NOEXCEPT
    {
        return this->m_debug_mode;
    }

    tjob_t::elem_by_elem_config_t const*
    tjob_t::ptrElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        return ( ( this->m_elem_by_elem_buffer.get() != nullptr ) &&
                 ( this->m_elem_by_elem_buffer->getNumObjects() >
                    st_size_t{ 0 } ) )
            ? this->m_elem_by_elem_buffer->get<
                tjob_t::elem_by_elem_config_t >( st_size_t{ 0 } )
            : nullptr;
    }

    bool
    tjob_t::elemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return ::NS(ElemByElemConfig_is_rolling)(
            this->ptrElemByElemConfig() );
    }

    bool
    tjob_t::defaultElemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return this->m_default_elem_by_elem_rolling;
    }

    void tjob_t::setDefaultElemByElemRolling(
        bool is_rolling ) SIXTRL_NOEXCEPT
    {
        this->m_default_elem_by_elem_rolling = is_rolling;
    }

    /* --------------------------------------------------------------------- */

    tjob_t::TrackJobBase(
        const char *const SIXTRL_RESTRICT type_str,
        track_job_type_t const type_id ) :
        m_type_str(),
        m_device_id_str(),
        m_config_str(),
        m_assign_address_items(),
        m_particle_set_indices(),
        m_num_particles_in_sets(),
        m_beam_monitor_indices(),
        m_stored_buffers(),
        m_my_output_buffer( nullptr ),
        m_elem_by_elem_buffer( nullptr ),
        m_particles_addr_buffer( nullptr ),
        m_ptr_particles_buffer( nullptr ),
        m_ptr_beam_elem_buffer( nullptr ),
        m_ptr_output_buffer( nullptr ),
        m_ptr_c_particles_buffer( nullptr ),
        m_ptr_c_beam_elem_buffer( nullptr ),
        m_ptr_c_output_buffer( nullptr ),
        m_be_mon_output_buffer_offset( st_size_t{ 0 } ),
        m_elem_by_elem_output_offset( st_size_t{ 0 } ),
        m_total_num_particles_in_sets( st_size_t{ 0 } ),
        m_num_stored_buffers( st_size_t{ 0 } ),
        m_type_id( type_id ),
        m_default_elem_by_elem_order( ::NS(ELEM_BY_ELEM_ORDER_DEFAULT) ),
        m_min_particle_id( tjob_t::particle_index_t{ 0 } ),
        m_max_particle_id( tjob_t::particle_index_t{ 0 } ),
        m_min_element_id( tjob_t::particle_index_t{ 0 } ),
        m_max_element_id( tjob_t::particle_index_t{ 0 } ),
        m_min_initial_turn_id( tjob_t::particle_index_t{ 0 } ),
        m_max_initial_turn_id( tjob_t::particle_index_t{ 0 } ),
        m_until_turn_elem_by_elem( st_size_t{ 0 } ),
        m_collect_flags(
            st::TRACK_JOB_IO_DEFAULT_FLAGS ),
        m_requires_collect( true ),
        m_default_elem_by_elem_rolling( true ),
        m_has_beam_monitor_output( false ),
        m_has_elem_by_elem_output( false ),
        m_debug_mode( false )
    {
        if( type_str != nullptr ) this->m_type_str = type_str;

        this->doInitDefaultParticleSetIndices();
        this->doInitDefaultBeamMonitorIndices();
    }

    tjob_t::TrackJobBase( TrackJobBase const& other ) :
        m_type_str( other.m_type_str ),
        m_device_id_str( other.m_type_str ),
        m_config_str( other.m_config_str ),
        m_assign_address_items( other.m_assign_address_items ),
        m_particle_set_indices( other.m_particle_set_indices ),
        m_num_particles_in_sets( other.m_num_particles_in_sets ),
        m_beam_monitor_indices( other.m_beam_monitor_indices ),
        m_stored_buffers( other.m_stored_buffers ),
        m_my_output_buffer( nullptr ),
        m_elem_by_elem_buffer( nullptr ),
        m_particles_addr_buffer( nullptr ),
        m_ptr_particles_buffer( other.m_ptr_particles_buffer  ),
        m_ptr_beam_elem_buffer( other.m_ptr_beam_elem_buffer ),
        m_ptr_output_buffer( nullptr ),
        m_ptr_c_particles_buffer( other.m_ptr_c_particles_buffer ),
        m_ptr_c_beam_elem_buffer( other.m_ptr_c_beam_elem_buffer ),
        m_ptr_c_output_buffer( nullptr ),
        m_be_mon_output_buffer_offset( other.m_be_mon_output_buffer_offset ),
        m_elem_by_elem_output_offset( other.m_elem_by_elem_output_offset ),
        m_total_num_particles_in_sets( other.m_total_num_particles_in_sets ),
        m_num_stored_buffers( other.m_num_stored_buffers ),
        m_type_id( other.m_type_id ),
        m_default_elem_by_elem_order( other.m_default_elem_by_elem_order ),
        m_min_particle_id( other.m_min_particle_id ),
        m_max_particle_id( other.m_max_particle_id ),
        m_min_element_id( other.m_min_element_id ),
        m_max_element_id( other.m_max_element_id ),
        m_min_initial_turn_id( other.m_min_initial_turn_id ),
        m_max_initial_turn_id( other.m_max_initial_turn_id ),
        m_until_turn_elem_by_elem( other.m_until_turn_elem_by_elem ),
        m_collect_flags( other.m_collect_flags ),
        m_requires_collect( other.m_requires_collect ),
        m_default_elem_by_elem_rolling( other.m_default_elem_by_elem_rolling ),
        m_has_beam_monitor_output( other.m_has_beam_monitor_output ),
        m_has_elem_by_elem_output( other.m_has_elem_by_elem_output ),
        m_debug_mode( other.m_debug_mode )
    {
        if( other.ownsOutputBuffer() )
        {
            tjob_t::COPY_PTR_BUFFER(
                this->m_my_output_buffer, other.m_my_output_buffer );

            this->m_ptr_output_buffer = this->m_my_output_buffer.get();
            if( this->m_ptr_output_buffer != nullptr )
            {
                this->m_ptr_c_output_buffer =
                    this->m_ptr_output_buffer->getCApiPtr();
            }
        }

        tjob_t::COPY_PTR_BUFFER(
            this->m_elem_by_elem_buffer, other.m_elem_by_elem_buffer );

        tjob_t::COPY_PTR_BUFFER(
            this->m_particles_addr_buffer, other.m_particles_addr_buffer );
    }

    tjob_t::TrackJobBase(
        TrackJobBase&& o ) SIXTRL_NOEXCEPT :
        m_type_str( std::move( o.m_type_str ) ),
        m_device_id_str( std::move( o.m_type_str ) ),
        m_config_str( std::move( o.m_config_str ) ),
        m_assign_address_items( std::move( o.m_assign_address_items ) ),
        m_particle_set_indices( std::move( o.m_particle_set_indices ) ),
        m_num_particles_in_sets( std::move( o.m_num_particles_in_sets ) ),
        m_beam_monitor_indices( std::move( o.m_beam_monitor_indices ) ),
        m_stored_buffers( std::move( o.m_stored_buffers ) ),
        m_my_output_buffer( std::move( o.m_my_output_buffer ) ),
        m_elem_by_elem_buffer( std::move( o.m_elem_by_elem_buffer ) ),
        m_particles_addr_buffer( std::move( o.m_particles_addr_buffer ) ),
        m_ptr_particles_buffer( std::move( o.m_ptr_particles_buffer  ) ),
        m_ptr_beam_elem_buffer( std::move( o.m_ptr_beam_elem_buffer ) ),
        m_ptr_output_buffer( std::move( o.m_ptr_output_buffer ) ),
        m_ptr_c_particles_buffer( std::move( o.m_ptr_c_particles_buffer ) ),
        m_ptr_c_beam_elem_buffer( std::move( o.m_ptr_c_beam_elem_buffer ) ),
        m_ptr_c_output_buffer( std::move( o.m_ptr_c_output_buffer ) ),
        m_be_mon_output_buffer_offset( std::move(
            o.m_be_mon_output_buffer_offset ) ),
        m_elem_by_elem_output_offset( std::move(
            o.m_elem_by_elem_output_offset ) ),
        m_total_num_particles_in_sets( std::move(
            o.m_total_num_particles_in_sets ) ),
        m_num_stored_buffers( std::move( o.m_num_stored_buffers ) ),
        m_type_id( std::move( o.m_type_id ) ),
        m_default_elem_by_elem_order( std::move(
            o.m_default_elem_by_elem_order ) ),
        m_min_particle_id( std::move( o.m_min_particle_id ) ),
        m_max_particle_id( std::move( o.m_max_particle_id ) ),
        m_min_element_id( std::move( o.m_min_element_id ) ),
        m_max_element_id( std::move( o.m_max_element_id ) ),
        m_min_initial_turn_id( std::move( o.m_min_initial_turn_id ) ),
        m_max_initial_turn_id( std::move( o.m_max_initial_turn_id ) ),
        m_until_turn_elem_by_elem( std::move( o.m_until_turn_elem_by_elem ) ),
        m_collect_flags( std::move( o.m_collect_flags ) ),
        m_requires_collect( std::move( o.m_requires_collect ) ),
        m_default_elem_by_elem_rolling( std::move(
            o.m_default_elem_by_elem_rolling ) ),
        m_has_beam_monitor_output( std::move( o.m_has_beam_monitor_output ) ),
        m_has_elem_by_elem_output( std::move( o.m_has_elem_by_elem_output ) ),
        m_debug_mode( std::move( o.m_debug_mode ) )
    {
        o.m_type_str.clear();
        o.m_device_id_str.clear();
        o.m_config_str.clear();

        o.doClearBaseImpl();
    }

    TrackJobBase& tjob_t::operator=(
        TrackJobBase const& rhs )
    {
        if( this != &rhs )
        {
            this->m_type_str                    = rhs.m_type_str;
            this->m_device_id_str               = rhs.m_type_str;
            this->m_config_str                  = rhs.m_config_str;
            this->m_particle_set_indices        = rhs.m_particle_set_indices;
            this->m_num_particles_in_sets       = rhs.m_num_particles_in_sets;
            this->m_beam_monitor_indices        = rhs.m_beam_monitor_indices;
            this->m_ptr_particles_buffer        = rhs.m_ptr_particles_buffer;
            this->m_ptr_beam_elem_buffer        = rhs.m_ptr_beam_elem_buffer;
            this->m_ptr_c_particles_buffer      = rhs.m_ptr_c_particles_buffer;
            this->m_ptr_c_beam_elem_buffer      = rhs.m_ptr_c_beam_elem_buffer;

            this->m_be_mon_output_buffer_offset =
                rhs.m_be_mon_output_buffer_offset;

            this->m_elem_by_elem_output_offset =
                rhs.m_elem_by_elem_output_offset;

            this->m_total_num_particles_in_sets =
                rhs.m_total_num_particles_in_sets;

            this->m_num_stored_buffers         = rhs.m_num_stored_buffers;
            this->m_default_elem_by_elem_order =
                rhs.m_default_elem_by_elem_order;

            this->m_type_id                    = rhs.m_type_id;
            this->m_min_particle_id            = rhs.m_min_particle_id;
            this->m_max_particle_id            = rhs.m_max_particle_id;
            this->m_min_element_id             = rhs.m_min_element_id;
            this->m_max_element_id             = rhs.m_max_element_id;
            this->m_min_initial_turn_id        = rhs.m_min_initial_turn_id;
            this->m_max_initial_turn_id        = rhs.m_max_initial_turn_id;
            this->m_until_turn_elem_by_elem    = rhs.m_until_turn_elem_by_elem;

            this->m_default_elem_by_elem_rolling =
                rhs.m_default_elem_by_elem_rolling;

            this->m_has_beam_monitor_output = rhs.m_has_beam_monitor_output;
            this->m_has_elem_by_elem_output = rhs.m_has_elem_by_elem_output;
            this->m_requires_collect        = rhs.m_requires_collect;
            this->m_collect_flags           = rhs.m_collect_flags;
            this->m_assign_address_items     = rhs.m_assign_address_items;
            this->m_stored_buffers          = rhs.m_stored_buffers;

            if( rhs.ownsOutputBuffer() )
            {
                tjob_t::COPY_PTR_BUFFER(
                    this->m_my_output_buffer, rhs.m_my_output_buffer );

               this->m_ptr_output_buffer = this->m_my_output_buffer.get();
               if( this->m_ptr_output_buffer != nullptr )
               {
                   this->m_ptr_c_output_buffer =
                    this->m_ptr_output_buffer->getCApiPtr();
               }
            }
            else
            {
                this->m_ptr_output_buffer   = rhs.m_ptr_output_buffer;
                this->m_ptr_c_output_buffer = rhs.m_ptr_c_output_buffer;
            }

            tjob_t::COPY_PTR_BUFFER(
                this->m_elem_by_elem_buffer, rhs.m_elem_by_elem_buffer );

            tjob_t::COPY_PTR_BUFFER(
                this->m_particles_addr_buffer, rhs.m_particles_addr_buffer );

            this->m_debug_mode = rhs.m_debug_mode;
        }

        return *this;
    }

    TrackJobBase& tjob_t::operator=(
        TrackJobBase&& rhs ) SIXTRL_NOEXCEPT
    {
        if( this != &rhs )
        {
            this->m_type_str         = std::move( rhs.m_type_str );
            this->m_device_id_str    = std::move( rhs.m_type_str );
            this->m_config_str       = std::move( rhs.m_config_str );

            this->m_particle_set_indices =
                std::move( rhs.m_particle_set_indices );

            this->m_num_particles_in_sets =
                std::move( rhs.m_num_particles_in_sets );

            this->m_beam_monitor_indices =
                std::move( rhs.m_beam_monitor_indices );

            this->m_stored_buffers   = std::move( rhs.m_stored_buffers );
            this->m_my_output_buffer = std::move( rhs.m_my_output_buffer );

            this->m_elem_by_elem_buffer =
                std::move( rhs.m_elem_by_elem_buffer );

            this->m_particles_addr_buffer =
                std::move( rhs.m_particles_addr_buffer );

            this->m_assign_address_items =
                std::move( rhs.m_assign_address_items );

            this->m_ptr_particles_buffer =
                std::move( rhs.m_ptr_particles_buffer );

            this->m_ptr_beam_elem_buffer =
                std::move( rhs.m_ptr_beam_elem_buffer );

            this->m_ptr_output_buffer = std::move( rhs.m_ptr_output_buffer );

            this->m_ptr_c_particles_buffer =
                std::move( rhs.m_ptr_c_particles_buffer );

            this->m_ptr_c_beam_elem_buffer =
                std::move( rhs.m_ptr_c_beam_elem_buffer );

            this->m_ptr_c_output_buffer =
                std::move( rhs.m_ptr_c_output_buffer );

            this->m_be_mon_output_buffer_offset =
                std::move( rhs.m_be_mon_output_buffer_offset );

            this->m_elem_by_elem_output_offset =
                std::move( rhs.m_elem_by_elem_output_offset );

            this->m_total_num_particles_in_sets =
                std::move( rhs.m_total_num_particles_in_sets );

            this->m_num_stored_buffers =
                std::move( rhs.m_num_stored_buffers );

            this->m_type_id = std::move( rhs.m_type_id );

            this->m_default_elem_by_elem_order =
                std::move( rhs.m_default_elem_by_elem_order );

            this->m_min_particle_id     = std::move( rhs.m_min_particle_id );
            this->m_max_particle_id     = std::move( rhs.m_max_particle_id );
            this->m_min_element_id      = std::move( rhs.m_min_element_id  );
            this->m_max_element_id      = std::move( rhs.m_max_element_id  );
            this->m_min_initial_turn_id = std::move( rhs.m_min_initial_turn_id);
            this->m_max_initial_turn_id = std::move( rhs.m_max_initial_turn_id);

            this->m_until_turn_elem_by_elem =
                std::move( rhs.m_until_turn_elem_by_elem );

            this->m_requires_collect = std::move( rhs.m_requires_collect );
            this->m_collect_flags = std::move( rhs.m_collect_flags );

            this->m_default_elem_by_elem_rolling =
                std::move( rhs.m_default_elem_by_elem_rolling );

            this->m_has_beam_monitor_output =
                std::move( rhs.m_has_beam_monitor_output );

            this->m_has_elem_by_elem_output =
                std::move( rhs.m_has_elem_by_elem_output );

            this->m_debug_mode = std::move( rhs.m_debug_mode );

            rhs.m_type_str.clear();
            rhs.m_device_id_str.clear();
            rhs.m_config_str.clear();

            rhs.m_type_id = type_t{ 0 };
            rhs.doClearBaseImpl();
        }

        return *this;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doClear()
    {
        this->doClearBaseImpl();
    }

    void tjob_t::doCollect( collect_flag_t const ) {}
    void tjob_t::doPush( push_flag_t const ) {}

    tjob_t::track_status_t tjob_t::doTrackUntilTurn( size_type const )
    {
        return tjob_t::track_status_t{ -1 };
    }

    tjob_t::track_status_t tjob_t::doTrackElemByElem( size_type const )
    {
        return tjob_t::track_status_t{ -1 };
    }

    tjob_t::track_status_t
    tjob_t::doTrackLine( size_type const, size_type const, bool const )
    {
        return tjob_t::track_status_t{ -1 };
    }

    /* --------------------------------------------------------------------- */

    bool tjob_t::doPrepareParticlesStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        bool success = false;
        using p_index_t = tjob_t::particle_index_t;

        SIXTRL_STATIC_VAR st_size_t const ZERO = st_size_t{ 0 };
        SIXTRL_STATIC_VAR st_size_t const ONE  = st_size_t{ 1 };

        st_size_t const nn = this->numParticleSets();
        st_size_t const num_psets = ::NS(Buffer_get_num_of_objects)( pb );

        if( ( pb != nullptr ) && ( ( nn > ZERO ) || ( nn == ZERO ) ) &&
            ( !::NS(Buffer_needs_remapping)( pb ) ) && ( num_psets >= nn ) )
        {
            int ret = int{ -1 };

            st_size_t const first_index = ( nn > ZERO )
                ? this->particleSetIndex( ZERO ) : ZERO;

            p_index_t min_part_id, max_part_id, min_elem_id, max_elem_id,
                      min_turn_id, max_turn_id;

            if( ( nn <= ONE ) && ( first_index == ZERO ) )
            {
                ret = ::NS(Particles_get_min_max_attributes)(
                        ::NS(Particles_buffer_get_const_particles)( pb, ZERO ),
                        &min_part_id, &max_part_id, &min_elem_id, &max_elem_id,
                        &min_turn_id, &max_turn_id );
            }
            else if( nn > ZERO )
            {
                ret =
                ::NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
                    pb, nn, this->particleSetIndicesBegin(),
                    &min_part_id, &max_part_id, &min_elem_id, &max_elem_id,
                    &min_turn_id, &max_turn_id );
            }

            if( ret == int{ 0 } )
            {
                this->doSetMinParticleId( min_part_id );
                this->doSetMaxParticleId( max_part_id );

                this->doSetMinElementId( min_elem_id );
                this->doSetMaxElementId( max_elem_id );

                this->doSetMinInitialTurnId( min_turn_id );
                this->doSetMaxInitialTurnId( max_turn_id );

                success = true;
            }

            if( success )
            {
                if( this->m_particles_addr_buffer.get() == nullptr )
                {
                    tjob_t::ptr_buffer_t particles_addr_buffer(
                        new tjob_t::buffer_t );

                    this->doUpdateStoredParticlesAddrBuffer(
                        std::move( particles_addr_buffer ) );
                }

                if( this->m_particles_addr_buffer.get() != nullptr )
                {
                    success = ( st::ARCH_STATUS_SUCCESS ==
                    ::NS(ParticlesAddr_prepare_buffer_based_on_particles_buffer)(
                        this->m_particles_addr_buffer->getCApiPtr(), pb ) );
                }
            }
        }

        return success;
    }

    bool tjob_t::doPrepareBeamElementsStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT belems )
    {
        bool success     = false;
        using p_index_t  = tjob_t::particle_index_t;
        using buf_size_t = ::NS(buffer_size_t);
        using obj_ptr_t  = SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object)*;
        using ptr_t      = SIXTRL_BE_ARGPTR_DEC NS(BeamMonitor)*;

        SIXTRL_STATIC_VAR st_size_t const ZERO = st_size_t{ 0 };
        SIXTRL_STATIC_VAR uintptr_t const UZERO = uintptr_t{ 0 };

        if( ( belems != nullptr ) &&
            ( !::NS(Buffer_needs_remapping)( belems ) ) &&
            ( ::NS(Buffer_get_num_of_objects)( belems ) > ZERO ) &&
            ( ::NS(BeamElements_is_beam_elements_buffer)( belems ) ) )
        {
            int ret = -1;

            p_index_t const start_be_id = p_index_t{ 0 };
            buf_size_t  num_e_by_e_objs = buf_size_t{ 0 };
            p_index_t min_elem_id = this->minElementId();
            p_index_t max_elem_id = this->maxElementId();

            ret = ::NS(ElemByElemConfig_find_min_max_element_id_from_buffer)(
                belems, &min_elem_id, &max_elem_id, &num_e_by_e_objs,
                    start_be_id );

            if( ret == 0 )
            {
                buf_size_t num_be_monitors = buf_size_t{ 0 };
                std::vector< st_size_t > be_mon_indices( num_e_by_e_objs, ZERO );

                num_be_monitors = ::NS(BeamMonitor_monitor_indices_from_buffer)(
                    be_mon_indices.data(), be_mon_indices.size(), belems );

                if( num_be_monitors <= be_mon_indices.size() )
                {
                    auto ind_end = be_mon_indices.begin();

                    if( num_be_monitors > buf_size_t{ 0 } )
                    {
                        std::advance( ind_end, num_be_monitors );
                    }

                    this->doSetBeamMonitorIndices(
                        be_mon_indices.begin(), ind_end );
                    SIXTRL_ASSERT( num_be_monitors == this->numBeamMonitors() );

                    this->doSetMinElementId( min_elem_id );
                    this->doSetMaxElementId( max_elem_id );
                }
                else
                {
                    ret = -1;
                }
            }

            success = ( ret == 0 );

            if( ( success ) && ( this->numBeamMonitors() > ZERO ) &&
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
                        ::NS(BeamMonitor_set_min_particle_id)( m, min_part_id );
                        ::NS(BeamMonitor_set_max_particle_id)( m, max_part_id );
                    }
                    else
                    {
                        success = false;
                        break;
                    }
                }
            }
        }

        return success;
    }

    bool tjob_t::doPrepareOutputStructures(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        bool success = false;

        using c_buffer_t = tjob_t::c_buffer_t;
        using elem_by_elem_config_t = tjob_t::elem_by_elem_config_t;
        using ptr_output_buffer_t = tjob_t::ptr_buffer_t;
        using par_index_t = tjob_t::particle_index_t;

        ( void )particles_buffer;

        SIXTRL_ASSERT( particles_buffer != nullptr );
        SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( particles_buffer ) );

        SIXTRL_ASSERT( ( ::NS(Buffer_get_num_of_objects)( particles_buffer ) ==
            st_size_t{ 0 } ) || ( ::NS(Buffer_is_particles_buffer)(
                particles_buffer ) ) );

        SIXTRL_ASSERT( beam_elements_buffer != nullptr );
        SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( beam_elements_buffer ) );

        SIXTRL_ASSERT( ::NS(Buffer_get_num_of_objects)( beam_elements_buffer )
            > st_size_t{ 0 } );

        SIXTRL_ASSERT( ::NS(BeamElements_is_beam_elements_buffer)(
            beam_elements_buffer ) );

        c_buffer_t* output_buffer = ptr_output_buffer;

        if( output_buffer == nullptr )
        {
            if( !this->hasOutputBuffer() )
            {
                SIXTRL_ASSERT( !this->ownsOutputBuffer() );
                tjob_t::ptr_buffer_t managed_output_buffer(
                    new tjob_t::buffer_t );

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
                return success;
            }
        }

        if( output_buffer != nullptr )
        {
            SIXTRL_STATIC_VAR const st_size_t ZERO = st_size_t{ 0 };
            SIXTRL_ASSERT( !::NS(Buffer_needs_remapping)( output_buffer ) );

            SIXTRL_ASSERT(
                ( ::NS(Buffer_get_num_of_objects)( output_buffer ) == ZERO )||
                ( ::NS(Buffer_is_particles_buffer)( output_buffer ) ) );

            st_size_t elem_by_elem_out_idx_offset = st_size_t{ 0 };
            st_size_t be_monitor_out_idx_offset   = st_size_t{ 0 };
            par_index_t max_elem_by_elem_turn_id  = par_index_t{ 0 };

            int ret = ::NS(OutputBuffer_prepare_detailed)(
                beam_elements_buffer, output_buffer,
                this->minParticleId(), this->maxParticleId(),
                this->minElementId(),  this->maxElementId(),
                this->minInitialTurnId(), this->maxInitialTurnId(),
                until_turn_elem_by_elem,
                &elem_by_elem_out_idx_offset, &be_monitor_out_idx_offset,
                &max_elem_by_elem_turn_id );

            if( this->m_elem_by_elem_buffer.get() == nullptr )
            {
                tjob_t::ptr_buffer_t elem_by_elem_buffer(
                    new tjob_t::buffer_t );

                this->doUpdateStoredElemByElemConfig(
                    std::move( elem_by_elem_buffer ) );
            }
            else
            {
                this->m_elem_by_elem_buffer->reset();
            }

            SIXTRL_ASSERT( this->m_elem_by_elem_buffer.get() != nullptr );
            SIXTRL_ASSERT( this->m_elem_by_elem_buffer->getNumObjects() ==
                           st_size_t{ 0 } );

            if( ( ret == 0 ) && ( until_turn_elem_by_elem > ZERO ) &&
                ( this->minInitialTurnId() >= par_index_t{ 0 } ) &&
                ( max_elem_by_elem_turn_id >= this->minInitialTurnId() ) &&
                ( until_turn_elem_by_elem > static_cast< st_size_t >(
                    this->minInitialTurnId() ) ) )
            {
                elem_by_elem_config_t* conf = ::NS(ElemByElemConfig_preset)(
                    ::NS(ElemByElemConfig_new)(
                        this->m_elem_by_elem_buffer->getCApiPtr() ) );

                SIXTRL_ASSERT( conf != nullptr );

                ret = ::NS(ElemByElemConfig_init_detailed)( conf,
                    this->defaultElemByElemOrder(),
                    this->minParticleId(), this->maxParticleId(),
                    this->minElementId(),  this->maxElementId(),
                    this->minInitialTurnId(), max_elem_by_elem_turn_id,
                    this->defaultElemByElemRolling() );

                if( ret == 0 )
                {
                    this->doSetUntilTurnElemByElem( until_turn_elem_by_elem );
                    SIXTRL_ASSERT( this->hasElemByElemConfig() );
                }
            }
            else if( ret == 0 )
            {
                this->doSetUntilTurnElemByElem( ZERO );
                ret = ( !this->hasElemByElemConfig() ) ? 0 : -1;
            }

            if( ret == 0 )
            {
                this->doSetBeamMonitorOutputBufferOffset(
                    be_monitor_out_idx_offset );

                this->doSetElemByElemOutputIndexOffset(
                    elem_by_elem_out_idx_offset );
            }

            success = ( ret == 0 );
        }

        return success;
    }

    bool tjob_t::doAssignOutputBufferToBeamMonitors(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        tjob_t::particle_index_t const min_turn_id,
        st_size_t const output_buffer_offset_index )
    {
        bool success = false;

        this->doSetBeamMonitorOutputEnabledFlag( false );

        if( ( output_buffer != nullptr ) && ( beam_elem_buffer != nullptr ) &&
            ( this->numBeamMonitors() > st_size_t{ 0 } ) &&
            ( min_turn_id >= tjob_t::particle_index_t{ 0 } ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
              output_buffer_offset_index ) )
        {
            if( !this->debugMode() )
            {
                success = ( st::ARCH_STATUS_SUCCESS ==
                    ::NS(BeamMonitor_assign_output_buffer_from_offset)(
                        beam_elem_buffer, output_buffer, min_turn_id,
                            output_buffer_offset_index ) );
            }
            else
            {
                st::arch_debugging_t status_flags =
                    st::ARCH_DEBUGGING_GENERAL_FAILURE;

                SIXTRL_ASSERT( ::NS(Buffer_get_slot_size)( output_buffer ) ==
                    ::NS(Buffer_get_slot_size)( beam_elem_buffer ) );

                success = ( st::ARCH_STATUS_SUCCESS ==
                    NS(BeamMonitor_assign_output_buffer_from_offset_debug)(
                        beam_elem_buffer, output_buffer, min_turn_id,
                            output_buffer_offset_index, &status_flags ) );

                if( success )
                {
                    success = ( st::ARCH_STATUS_SUCCESS ==
                        ::NS(DebugReg_get_stored_arch_status)( status_flags ) );
                }
            }

            if( success )
            {
                this->doSetBeamMonitorOutputEnabledFlag( true );
            }
        }

        return success;
    }

    bool tjob_t::doAssignOutputBufferToElemByElemConfig(
        tjob_t::elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        st_size_t const output_buffer_offset_index )
    {
        bool success = false;

        if( ( elem_by_elem_config != nullptr ) && ( output_buffer != nullptr ) &&
            ( ::NS(Buffer_get_num_of_objects)( output_buffer ) >
                output_buffer_offset_index ) )
        {
            this->doSetElemByElemOutputEnabledFlag( false );

            if( this->debugMode() )
            {
                success = ( st::ARCH_STATUS_SUCCESS ==
                    ::NS(ElemByElemConfig_assign_output_buffer)(
                        elem_by_elem_config, output_buffer,
                            output_buffer_offset_index ) );
            }
            else
            {
                st::arch_debugging_t status_flags =
                    st::ARCH_DEBUGGING_GENERAL_FAILURE;

                success = ( st::ARCH_STATUS_SUCCESS ==
                    ::NS(ElemByElemConfig_assign_output_buffer_debug)(
                        elem_by_elem_config, output_buffer,
                            output_buffer_offset_index, &status_flags ) );

                if( success )
                {
                    success = ( st::ARCH_STATUS_SUCCESS ==
                        ::NS(DebugReg_get_stored_arch_status)( status_flags ) );
                }
            }

            if( success )
            {
                this->doSetElemByElemOutputEnabledFlag( true );
            }
        }

        return success;
    }

    bool tjob_t::doReset(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        tjob_t::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        st_size_t const until_turn_elem_by_elem )
    {
        using output_buffer_flag_t = tjob_t::output_buffer_flag_t;

        bool success = this->doPrepareParticlesStructures( particles_buffer );

        if( success )
        {
            success = this->doPrepareBeamElementsStructures( beam_elem_buffer );
        }

        if( success )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
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
                success = this->doPrepareOutputStructures( particles_buffer,
                    beam_elem_buffer, output_buffer, until_turn_elem_by_elem );
            }

            if( ( success ) && ( this->hasOutputBuffer() ) &&
                ( requires_output_buffer ) )
            {
                if( ::NS(OutputBuffer_requires_elem_by_elem_output)(
                        out_buffer_flags ) )
                {
                    success = this->doAssignOutputBufferToElemByElemConfig(
                        this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                            this->elemByElemOutputBufferOffset() );
                }

                if( ( success ) &&
                    ( ::NS(OutputBuffer_requires_beam_monitor_output)(
                        out_buffer_flags ) ) )
                {
                    success = this->doAssignOutputBufferToBeamMonitors(
                        beam_elem_buffer, this->ptrCOutputBuffer(),
                        this->minInitialTurnId(),
                        this->beamMonitorsOutputBufferOffset() );
                }
            }

            if( ( success ) && ( output_buffer != nullptr ) &&
                ( !this->hasOutputBuffer() ) )
            {
                this->doSetPtrCOutputBuffer( output_buffer );
            }
        }

        return success;
    }

    bool tjob_t::doAssignNewOutputBuffer(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        bool success = false;
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
            success = this->doPrepareOutputStructures(
                this->ptrCParticlesBuffer(), this->ptrCBeamElementsBuffer(),
                ptr_output_buffer, this->numElemByElemTurns() );
        }

        if( ( success ) && ( requires_output_buffer ) &&
            ( this->hasOutputBuffer() ) )
        {
            if( ( ::NS(OutputBuffer_requires_beam_monitor_output)( flags ) ) &&
                ( this->hasBeamMonitorOutput() ) )
            {
                success = this->doAssignOutputBufferToBeamMonitors(
                    this->ptrCBeamElementsBuffer(), this->ptrCOutputBuffer(),
                    this->minInitialTurnId(),
                    this->beamMonitorsOutputBufferOffset() );
            }

            if( ( success ) &&
                ( ::NS(OutputBuffer_requires_elem_by_elem_output)( flags ) ) &&
                ( this->hasElemByElemOutput() ) )
            {
                success = this->doAssignOutputBufferToElemByElemConfig(
                    this->ptrElemByElemConfig(), this->ptrCOutputBuffer(),
                    this->elemByElemOutputBufferOffset() );
            }
        }

        return success;
    }

    st_size_t tjob_t::doAddStoredBuffer(
        tjob_t::buffer_store_t&& buffer_store_handle )
    {
        st_size_t buffer_index = st::ARCH_MIN_USER_DEFINED_BUFFER_ID +
            this->m_stored_buffers.size();

        tjob_t::c_buffer_t* ptr_cbuffer_null = nullptr;

        if( buffer_store_handle.active() )
        {
            ++this->m_num_stored_buffers;
        }

        this->m_stored_buffers.emplace_back( ptr_cbuffer_null, false );
        this->m_stored_buffers.back() = std::move( buffer_store_handle );

        return buffer_index;
    }

    st_status_t tjob_t::doRemoveStoredBuffer(
        st_size_t const buffer_index )
    {
        st_status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
        st_size_t const min_buffer_id = this->min_stored_buffer_id();
        st_size_t const max_buffer_id_plus_one =
            min_buffer_id + this->m_stored_buffers.size();

        if( ( min_buffer_id != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_index != st::ARCH_ILLEGAL_BUFFER_ID ) &&
            ( buffer_index >= min_buffer_id ) &&
            ( buffer_index < max_buffer_id_plus_one ) )
        {
            st_size_t const ii = buffer_index - min_buffer_id;

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

    st_status_t tjob_t::doPushStoredBuffer(
        st_size_t const buffer_id )
    {
        tjob_t::buffer_store_t const* ptr_stored_buffer =
            this->doGetPtrBufferStore( buffer_id );

        return ( ptr_stored_buffer != nullptr )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::doCollectStoredBuffer(
        st_size_t const buffer_id )
    {
        tjob_t::buffer_store_t const* ptr_stored_buffer =
            this->doGetPtrBufferStore( buffer_id );

        return ( ptr_stored_buffer != nullptr )
            ? st::ARCH_STATUS_SUCCESS : st::ARCH_STATUS_GENERAL_FAILURE;
    }

    st_status_t tjob_t::doAddAssignAddressItem(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF assign_item,
        st_size_t* SIXTRL_RESTRICT ptr_item_index )
    {
        using size_t = st_size_t;
        using buffer_t = tjob_t::buffer_t;
        using key_t = tjob_t::assign_item_key_t;

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
                item_index = this->doFindAssingAddressItem( assign_item );

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

    st_status_t tjob_t::doRemoveAssignAddressItem(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key,
        st_size_t const index_of_item_to_remove )
    {
        using size_t = st_size_t;
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

    st_status_t tjob_t::doPerformAddressAssignments(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key )
    {
        using size_t = st_size_t;
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

            status = ::NS(AssignAddressItem_perform_address_assignment_kernel_impl)(
                it->second.dataBegin< unsigned char const* >(),
                it->second.getSlotSize(), size_t{ 0 }, size_t{ 1 },
                dest_buffer_begin, dest_slot_size, dest_buffer_id,
                src_buffer_begin, src_slot_size, src_buffer_id );
        }

        return status;
    }

    st_status_t tjob_t::doRebuildAssignItemsBufferArg()
    {
        return st::ARCH_STATUS_SUCCESS;
    }

    st_status_t tjob_t::doCommitAddressAssignments()
    {
        return st::ARCH_STATUS_SUCCESS;
    }

    /* --------------------------------------------------------------------- */

    void tjob_t::doParseConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        this->doParseConfigStrBaseImpl( config_str );
    }

    void tjob_t::doSetDeviceIdStr(
        const char *const SIXTRL_RESTRICT device_id_str )
    {
        if( device_id_str != nullptr )
        {
            this->m_device_id_str = device_id_str;
        }
        else
        {
            this->m_device_id_str.clear();
        }
    }

    void tjob_t::doSetConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        if( config_str != nullptr )
        {
            this->m_config_str = config_str;
        }
        else
        {
            this->m_config_str.clear();
        }
    }

    void tjob_t::doSetPtrParticleBuffer(
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

    void tjob_t::doSetPtrCParticleBuffer(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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

    void tjob_t::doSetPtrCOutputBuffer(
        tjob_t::c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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
        st_size_t const output_buffer_offset ) SIXTRL_NOEXCEPT
    {
        this->m_be_mon_output_buffer_offset = output_buffer_offset;
    }

    void tjob_t::doSetUntilTurnElemByElem(
        tjob_t::particle_index_t const
            until_turn_elem_by_elem ) SIXTRL_NOEXCEPT
    {
        this->m_until_turn_elem_by_elem = until_turn_elem_by_elem;
    }

    void tjob_t::doSetElemByElemOutputIndexOffset(
        st_size_t const elem_by_elem_output_offset ) SIXTRL_NOEXCEPT
    {
        this->m_elem_by_elem_output_offset = elem_by_elem_output_offset;
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
        bool const elem_by_elem_flag ) SIXTRL_NOEXCEPT
    {
        this->m_has_elem_by_elem_output = elem_by_elem_flag;
    }

    void tjob_t::doInitDefaultParticleSetIndices()
    {
        this->m_particle_set_indices.clear();
        this->m_particle_set_indices.push_back( st_size_t{ 0 } );

        this->m_num_particles_in_sets.clear();
        this->m_num_particles_in_sets.push_back( st_size_t{ 0 } );
    }

    void tjob_t::doInitDefaultBeamMonitorIndices()
    {
        this->m_beam_monitor_indices.clear();
    }

    void tjob_t::doSetMinParticleId(
        tjob_t::particle_index_t const min_particle_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_particle_id = min_particle_id;
    }

    void tjob_t::doSetMaxParticleId(
        tjob_t::particle_index_t const max_particle_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_particle_id = max_particle_id;
    }

    void tjob_t::doSetMinElementId(
        tjob_t::particle_index_t const min_element_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_element_id = min_element_id;
    }

    void tjob_t::doSetMaxElementId(
        tjob_t::particle_index_t const max_element_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_element_id = max_element_id;
    }

    void tjob_t::doSetMinInitialTurnId(
        tjob_t::particle_index_t const min_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_initial_turn_id = min_turn_id;
    }

    void tjob_t::doSetMaxInitialTurnId(
        tjob_t::particle_index_t const max_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_initial_turn_id = max_turn_id;
    }

    void tjob_t::doUpdateStoredOutputBuffer(
        tjob_t::ptr_buffer_t&& ptr_output_buffer ) SIXTRL_NOEXCEPT
    {
        this->doSetPtrOutputBuffer( ptr_output_buffer.get() );
        this->m_my_output_buffer = std::move( ptr_output_buffer );
    }

    void tjob_t::doUpdateStoredElemByElemConfig(
        tjob_t::ptr_buffer_t&& ptr_elem_by_elem_buffer ) SIXTRL_NOEXCEPT
    {
        this->m_elem_by_elem_buffer = std::move( ptr_elem_by_elem_buffer );
    }

    void tjob_t::doUpdateStoredParticlesAddrBuffer(
        tjob_t::ptr_buffer_t&& ptr_particles_addr_buffer ) SIXTRL_NOEXCEPT
    {
        this->m_particles_addr_buffer = std::move( ptr_particles_addr_buffer );
    }

    tjob_t::buffer_store_t* tjob_t::doGetPtrBufferStore(
        st_size_t const buffer_id ) SIXTRL_NOEXCEPT
    {
        return const_cast< tjob_t::buffer_store_t* >( static_cast<
            tjob_t const& >( *this ).doGetPtrBufferStore( buffer_id ) );
    }

    tjob_t::buffer_store_t const* tjob_t::doGetPtrBufferStore(
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
            SIXTRL_ASSERT( this->doGetStoredBufferSize() >=
                           this->m_num_stored_buffers );

            ptr_buffer_store = &this->m_stored_buffers[
                buffer_id - min_buffer_id ];
        }

        return ptr_buffer_store;
    }

    st_size_t
    tjob_t::doGetStoredBufferSize() const SIXTRL_NOEXCEPT
    {
        return this->m_stored_buffers.size();
    }

    st_size_t tjob_t::doFindAssingAddressItem(
        tjob_t::assign_item_t const& SIXTRL_RESTRICT_REF
            item_to_search ) const SIXTRL_NOEXCEPT
    {
        using size_t = st_size_t;
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

    tjob_t::assign_item_t const* tjob_t::doGetAssignAddressItem(
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

    tjob_t::assign_item_t* tjob_t::doGetAssignAddressItem(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF key,
        st_size_t const item_index ) SIXTRL_NOEXCEPT
    {
        using ptr_t = tjob_t::assign_item_t*;
        return const_cast< ptr_t >( static_cast< tjob_t const& >(
            *this ).doGetAssignAddressItem( key, item_index ) );
    }

    tjob_t::c_buffer_t* tjob_t::doGetPtrAssignAddressItemsBuffer(
        tjob_t::assign_item_key_t const& SIXTRL_RESTRICT_REF
            key ) SIXTRL_NOEXCEPT
    {
        tjob_t::c_buffer_t* ptr_assign_address_items_buffer = nullptr;
        auto it = this->m_assign_address_items.find( key );

        if( it != this->m_assign_address_items.end() )
        {
            ptr_assign_address_items_buffer = it->second.getCApiPtr();
        }

        return ptr_assign_address_items_buffer;
    }

    void tjob_t::doClearBaseImpl() SIXTRL_NOEXCEPT
    {
        this->doInitDefaultParticleSetIndices();
        this->doInitDefaultBeamMonitorIndices();


        this->m_my_output_buffer.reset( nullptr );
        this->m_stored_buffers.clear();
        this->m_assign_address_items.clear();

        this->m_elem_by_elem_buffer.reset( nullptr );
        this->m_particles_addr_buffer.reset( nullptr );

        this->m_ptr_particles_buffer         = nullptr;
        this->m_ptr_beam_elem_buffer         = nullptr;
        this->m_ptr_output_buffer            = nullptr;

        this->m_ptr_c_particles_buffer       = nullptr;
        this->m_ptr_c_beam_elem_buffer       = nullptr;
        this->m_ptr_c_output_buffer          = nullptr;

        this->m_be_mon_output_buffer_offset  = st_size_t{ 0 };
        this->m_elem_by_elem_output_offset   = st_size_t{ 0 };
        this->m_default_elem_by_elem_order   = ::NS(ELEM_BY_ELEM_ORDER_DEFAULT);
        this->m_num_stored_buffers           = st_size_t{ 0 };

        ::NS(Particles_init_min_max_attributes_for_find)(
            &this->m_min_particle_id, &this->m_max_particle_id,
            &this->m_min_element_id,  &this->m_max_element_id,
            &this->m_min_initial_turn_id,
            &this->m_max_initial_turn_id );

        this->m_until_turn_elem_by_elem      = st_size_t{ 0 };

        this->m_default_elem_by_elem_rolling = true;
        this->m_has_beam_monitor_output      = false;
        this->m_has_elem_by_elem_output      = false;
    }

    void tjob_t::doParseConfigStrBaseImpl(
        const char *const SIXTRL_RESTRICT config_str )
    {
        ( void )config_str;
    }
}

#endif /* defined( __cplusplus ) */
/* end: sixtracklib/common/internal/track_job_base.cpp */
