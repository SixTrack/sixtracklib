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
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
    #include "sixtracklib/common/context/context_abs_base.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/output/output_buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_HOST_FN void TrackJobBase::clear()
    {
        this->doClear();
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::collect()
    {
        this->doCollect();
        return;
    }

    SIXTRL_HOST_FN TrackJobBase::track_status_t TrackJobBase::track(
        TrackJobBase::size_type const until_turn )
    {
        return this->doTrackUntilTurn( until_turn );
    }

    SIXTRL_HOST_FN TrackJobBase::track_status_t TrackJobBase::trackElemByElem(
        TrackJobBase::size_type const until_turn )
    {
        return this->doTrackElemByElem( until_turn );
    }

    SIXTRL_HOST_FN bool TrackJobBase::reset(
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF be_buffer,
        TrackJobBase::buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobBase::size_type const dump_elem_by_elem_turns )
    {
        using c_buffer_t = TrackJobBase::c_buffer_t;

        this->doClear();

        c_buffer_t* ptr_pb  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_eb  = be_buffer.getCApiPtr();
        c_buffer_t* ptr_out = ( ptr_output_buffer != nullptr ) ?
            ptr_output_buffer->getCApiPtr() : nullptr;

        bool const success = this->doReset(
            ptr_pb, ptr_eb, ptr_out, dump_elem_by_elem_turns );

        if( success )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &be_buffer );

            if( ( ptr_out != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobBase::reset(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobBase::size_type const dump_elem_by_elem_turns )
    {
        this->doClear();

        bool const success = this->doReset( particles_buffer, be_buffer,
                ptr_output_buffer, dump_elem_by_elem_turns );

        if( success )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( be_buffer );

            if( ( ptr_output_buffer != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrCOutputBuffer( ptr_output_buffer );
            }
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobBase::reset(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::size_type const num_particle_sets,
        TrackJobBase::size_type const* SIXTRL_RESTRICT
            particle_set_indices_begin,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT be_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobBase::size_type const dump_elem_by_elem_turns )
    {
        TrackJobBase::size_type const* particle_set_indices_end =
            particle_set_indices_begin;

        if( ( particle_set_indices_end != nullptr ) &&
            ( num_particle_sets > TrackJobBase::size_type{ 0 } ) )
        {
            std::advance( particle_set_indices_end, num_particle_sets );
        }

        return this->reset( particles_buffer,
            particle_set_indices_begin, particle_set_indices_end,
            be_buffer, ptr_output_buffer, dump_elem_by_elem_turns );
    }

    SIXTRL_HOST_FN bool TrackJobBase::assignOutputBuffer(
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF output_buffer )
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

    SIXTRL_HOST_FN bool TrackJobBase::assignOutputBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        return this->doAssignNewOutputBuffer( ptr_output_buffer );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN TrackJobBase::type_t
    TrackJobBase::type() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id;
    }

    SIXTRL_HOST_FN std::string const&
    TrackJobBase::typeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_str;
    }

    SIXTRL_HOST_FN char const* TrackJobBase::ptrTypeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_str.c_str();
    }

    SIXTRL_HOST_FN bool TrackJobBase::hasDeviceIdStr() const SIXTRL_RESTRICT
    {
        return ( !this->m_device_id_str.empty() );
    }

    SIXTRL_HOST_FN std::string const&
    TrackJobBase::deviceIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_device_id_str;
    }

    SIXTRL_HOST_FN char const*
    TrackJobBase::ptrDeviceIdStr() const SIXTRL_NOEXCEPT
    {
        return this->m_device_id_str.c_str();
    }

    SIXTRL_HOST_FN bool
    TrackJobBase::hasConfigStr() const SIXTRL_NOEXCEPT
    {
        return ( !this->m_config_str.empty() );
    }

    SIXTRL_HOST_FN std::string const&
    TrackJobBase::configStr()   const SIXTRL_NOEXCEPT
    {
        return this->m_config_str;
    }

    SIXTRL_HOST_FN char const*
    TrackJobBase::ptrConfigStr() const SIXTRL_NOEXCEPT
    {
        return this->m_config_str.c_str();
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN TrackJobBase::size_type
    TrackJobBase::numParticleSets() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_indices.size();
    }

    SIXTRL_HOST_FN TrackJobBase::size_type const*
    TrackJobBase::particleSetIndicesBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_particle_set_indices.data();
    }

    SIXTRL_HOST_FN TrackJobBase::size_type const*
    TrackJobBase::particleSetIndicesEnd() const SIXTRL_NOEXCEPT
    {
        TrackJobBase::size_type const* end_ptr =
            this->particleSetIndicesBegin();

        SIXTRL_ASSERT( end_ptr != nullptr );
        std::advance( end_ptr, this->numParticleSets() );

        return end_ptr;
    }

    SIXTRL_HOST_FN TrackJobBase::size_type TrackJobBase::particleSetIndex(
        TrackJobBase::size_type const n ) const
    {
        return this->m_particle_set_indices.at( n );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN TrackJobBase::particle_index_t
    TrackJobBase::minParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_particle_id;
    }

    SIXTRL_HOST_FN TrackJobBase::particle_index_t
    TrackJobBase::maxParticleId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_particle_id;
    }

    SIXTRL_HOST_FN TrackJobBase::particle_index_t
    TrackJobBase::minElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_min_element_id;
    }

    SIXTRL_HOST_FN TrackJobBase::particle_index_t
    TrackJobBase::maxElementId()  const SIXTRL_NOEXCEPT
    {
        return this->m_max_element_id;
    }

    SIXTRL_HOST_FN TrackJobBase::particle_index_t
    TrackJobBase::minInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_min_initial_turn_id;
    }

    SIXTRL_HOST_FN TrackJobBase::particle_index_t
    TrackJobBase::maxInitialTurnId() const SIXTRL_NOEXCEPT
    {
        return this->m_max_initial_turn_id;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN TrackJobBase::buffer_t*
    TrackJobBase::ptrParticlesBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = TrackJobBase;
        using ptr_t   = _this_t::buffer_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrParticlesBuffer() );
    }

    SIXTRL_HOST_FN TrackJobBase::buffer_t const*
    TrackJobBase::ptrParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_particles_buffer;
    }

    SIXTRL_HOST_FN TrackJobBase::c_buffer_t*
    TrackJobBase::ptrCParticlesBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = TrackJobBase;
        using ptr_t   = _this_t::c_buffer_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrCParticlesBuffer() );
    }

    SIXTRL_HOST_FN TrackJobBase::c_buffer_t const*
    TrackJobBase::ptrCParticlesBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_particles_buffer == nullptr ) ||
            ( this->m_ptr_particles_buffer->getCApiPtr() ==
              this->m_ptr_c_particles_buffer ) );

        return this->m_ptr_c_particles_buffer;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN TrackJobBase::buffer_t*
    TrackJobBase::ptrBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = TrackJobBase;
        using ptr_t   = _this_t::buffer_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrBeamElementsBuffer() );
    }

    SIXTRL_HOST_FN TrackJobBase::buffer_t const*
    TrackJobBase::ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_beam_elem_buffer;
    }

    SIXTRL_HOST_FN TrackJobBase::c_buffer_t*
    TrackJobBase::ptrCBeamElementsBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = TrackJobBase;
        using ptr_t   = _this_t::c_buffer_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrCBeamElementsBuffer() );
    }

    SIXTRL_HOST_FN TrackJobBase::c_buffer_t const*
    TrackJobBase::ptrCBeamElementsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_beam_elem_buffer == nullptr ) ||
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() ==
              this->m_ptr_c_beam_elem_buffer ) );

        return this->m_ptr_c_beam_elem_buffer;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN bool TrackJobBase::hasOutputBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->ptrCOutputBuffer() != nullptr );
    }

    SIXTRL_HOST_FN bool TrackJobBase::ownsOutputBuffer() const SIXTRL_NOEXCEPT
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

    SIXTRL_HOST_FN bool
    TrackJobBase::hasElemByElemOutput() const SIXTRL_NOEXCEPT
    {
        return this->m_has_elem_by_elem_output;
    }

    SIXTRL_HOST_FN bool
    TrackJobBase::hasBeamMonitorOutput() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            (  !this->m_has_beam_monitor_output ) ||
            ( ( this->m_has_beam_monitor_output ) &&
              ( this->m_ptr_c_output_buffer != nullptr ) ) );

        return this->m_has_beam_monitor_output;
    }

    SIXTRL_HOST_FN TrackJobBase::size_type
    TrackJobBase::beamMonitorsOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_be_mon_output_buffer_offset ) ) ||
            ( this->m_be_mon_output_buffer_offset ==
                TrackJobBase::size_type{ 0 } ) );

        return this->m_be_mon_output_buffer_offset;
    }

    SIXTRL_HOST_FN TrackJobBase::size_type
    TrackJobBase::elemByElemOutputBufferOffset() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( ( this->hasOutputBuffer() ) &&
              ( ::NS(Buffer_get_size)( this->ptrCOutputBuffer() ) >
                this->m_elem_by_elem_output_offset ) ) ||
            ( this->m_elem_by_elem_output_offset ==
                TrackJobBase::size_type{ 0 } ) );

        return this->m_elem_by_elem_output_offset;
    }

    SIXTRL_HOST_FN TrackJobBase::size_type
    TrackJobBase::numElemByElemTurns() const SIXTRL_NOEXCEPT
    {
        return this->m_dump_elem_by_elem_turns;
    }

    SIXTRL_HOST_FN TrackJobBase::buffer_t*
    TrackJobBase::ptrOutputBuffer() SIXTRL_RESTRICT
    {
        using _this_t = TrackJobBase;
        using ptr_t   = _this_t::buffer_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrOutputBuffer() );
    }

    SIXTRL_HOST_FN TrackJobBase::buffer_t*
    TrackJobBase::ptrOutputBuffer() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_output_buffer == nullptr ) ||
            ( this->m_ptr_output_buffer->getCApiPtr() ==
              this->m_ptr_c_output_buffer ) );

        return this->m_ptr_output_buffer;
    }

    SIXTRL_HOST_FN TrackJobBase::c_buffer_t*
    TrackJobBase::ptrCOutputBuffer() SIXTRL_RESTRICT
    {
        using _this_t = TrackJobBase;
        using ptr_t   = _this_t::c_buffer_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrCOutputBuffer() );
    }

    SIXTRL_HOST_FN TrackJobBase::c_buffer_t const*
    TrackJobBase::ptrCOutputBuffer() const SIXTRL_RESTRICT
    {
        SIXTRL_ASSERT(
            ( this->m_ptr_output_buffer == nullptr ) ||
            ( this->m_ptr_output_buffer->getCApiPtr() ==
              this->m_ptr_c_output_buffer ) );

        return this->m_ptr_c_output_buffer;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN bool TrackJobBase::hasBeamMonitors() const SIXTRL_NOEXCEPT
    {
        return !this->m_beam_monitor_indices.empty();
    }

    SIXTRL_HOST_FN TrackJobBase::size_type
    TrackJobBase::numBeamMonitors() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_indices.size();
    }


    SIXTRL_HOST_FN TrackJobBase::size_type const*
    TrackJobBase::beamMonitorIndicesBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_beam_monitor_indices.data();
    }

    SIXTRL_HOST_FN TrackJobBase::size_type const*
    TrackJobBase::beamMonitorIndicesEnd() const SIXTRL_NOEXCEPT
    {
        TrackJobBase::size_type const* end_ptr =
            this->beamMonitorIndicesBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->numBeamMonitors() );
        }

        return end_ptr;
    }

    SIXTRL_HOST_FN TrackJobBase::size_type
    TrackJobBase::beamMonitorIndex( TrackJobBase::size_type const n ) const
    {
        return this->m_beam_monitor_indices.at( n );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN bool
    TrackJobBase::hasElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( this->m_my_elem_by_elem_config.get() == nullptr ) ||
            ( this->m_ptr_c_output_buffer != nullptr ) );

        return ( ::NS(ElemByElemConfig_is_active)(
            this->m_my_elem_by_elem_config.get() ) );
    }

    SIXTRL_HOST_FN TrackJobBase::elem_by_elem_config_t*
    TrackJobBase::ptrElemByElemConfig() SIXTRL_NOEXCEPT
    {
        using _this_t = TrackJobBase;
        using  ptr_t  = _this_t::elem_by_elem_config_t*;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).ptrElemByElemConfig() );
    }

    SIXTRL_HOST_FN TrackJobBase::elem_by_elem_order_t
    TrackJobBase::elemByElemOrder() const SIXTRL_NOEXCEPT
    {
        return ::NS(ElemByElemConfig_get_order)(
            this->m_my_elem_by_elem_config.get() );
    }

    SIXTRL_HOST_FN TrackJobBase::elem_by_elem_order_t
    TrackJobBase::defaultElemByElemOrder() const SIXTRL_NOEXCEPT
    {
        return this->m_default_elem_by_elem_order;
    }

    SIXTRL_HOST_FN void TrackJobBase::setDefaultElemByElemOrder(
        TrackJobBase::elem_by_elem_order_t const order ) SIXTRL_NOEXCEPT
    {
        this->m_default_elem_by_elem_order = order;
        return;
    }

    SIXTRL_HOST_FN TrackJobBase::elem_by_elem_config_t const*
    TrackJobBase::ptrElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        return this->m_my_elem_by_elem_config.get();
    }

    SIXTRL_HOST_FN bool
    TrackJobBase::elemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return ::NS(ElemByElemConfig_is_rolling)(
            this->ptrElemByElemConfig() );
    }

    SIXTRL_HOST_FN bool
    TrackJobBase::defaultElemByElemRolling() const SIXTRL_NOEXCEPT
    {
        return this->m_default_elem_by_elem_rolling;
    }

    SIXTRL_HOST_FN void TrackJobBase::setDefaultElemByElemRolling(
        bool is_rolling ) SIXTRL_NOEXCEPT
    {
        this->m_default_elem_by_elem_rolling = is_rolling;
        return;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN TrackJobBase::TrackJobBase(
        const char *const SIXTRL_RESTRICT type_str,
        track_job_type_t const type_id ) :
        m_type_str(),
        m_device_id_str(),
        m_config_str(),
        m_particle_set_indices(),
        m_beam_monitor_indices(),
        m_my_output_buffer( nullptr ),
        m_my_elem_by_elem_config( nullptr ),
        m_ptr_particles_buffer( nullptr ),
        m_ptr_beam_elem_buffer( nullptr ),
        m_ptr_output_buffer( nullptr ),
        m_ptr_c_particles_buffer( nullptr ),
        m_ptr_c_beam_elem_buffer( nullptr ),
        m_ptr_c_output_buffer( nullptr ),
        m_be_mon_output_buffer_offset( TrackJobBase::size_type{ 0 } ),
        m_elem_by_elem_output_offset( TrackJobBase::size_type{ 0 } ),
        m_dump_elem_by_elem_turns( TrackJobBase::size_type{ 0 } ),
        m_type_id( type_id ),
        m_default_elem_by_elem_order( ::NS(ELEM_BY_ELEM_ORDER_DEFAULT) ),
        m_min_particle_id( TrackJobBase::particle_index_t{ 0 } ),
        m_max_particle_id( TrackJobBase::particle_index_t{ 0 } ),
        m_min_element_id( TrackJobBase::particle_index_t{ 0 } ),
        m_max_element_id( TrackJobBase::particle_index_t{ 0 } ),
        m_min_initial_turn_id( TrackJobBase::particle_index_t{ 0 } ),
        m_max_initial_turn_id( TrackJobBase::particle_index_t{ 0 } ),
        m_default_elem_by_elem_rolling( true ),
        m_has_beam_monitor_output( false ),
        m_has_elem_by_elem_output( false )
    {
        if( type_str != nullptr )
        {
            this->m_type_str = type_str;
        }

        this->doInitDefaultParticleSetIndices();
        this->doInitDefaultBeamMonitorIndices();
    }

    SIXTRL_HOST_FN TrackJobBase::TrackJobBase( TrackJobBase const& other ) :
        m_type_str( other.m_type_str ),
        m_device_id_str( other.m_type_str ),
        m_config_str( other.m_config_str ),
        m_particle_set_indices( other.m_particle_set_indices ),
        m_beam_monitor_indices( other.m_beam_monitor_indices ),
        m_my_output_buffer( nullptr  ),
        m_my_elem_by_elem_config( nullptr ),
        m_ptr_particles_buffer( other.m_ptr_particles_buffer  ),
        m_ptr_beam_elem_buffer( other.m_ptr_beam_elem_buffer ),
        m_ptr_output_buffer( nullptr ),
        m_ptr_c_particles_buffer( other.m_ptr_c_particles_buffer ),
        m_ptr_c_beam_elem_buffer( other.m_ptr_c_beam_elem_buffer ),
        m_ptr_c_output_buffer( nullptr ),
        m_be_mon_output_buffer_offset( other.m_be_mon_output_buffer_offset ),
        m_elem_by_elem_output_offset( other.m_elem_by_elem_output_offset ),
        m_dump_elem_by_elem_turns( other.m_dump_elem_by_elem_turns ),
        m_type_id( other.m_type_id ),
        m_default_elem_by_elem_order( other.m_default_elem_by_elem_order ),
        m_min_particle_id( other.m_min_particle_id ),
        m_max_particle_id( other.m_max_particle_id ),
        m_min_element_id( other.m_min_element_id ),
        m_max_element_id( other.m_max_element_id ),
        m_min_initial_turn_id( other.m_min_initial_turn_id ),
        m_max_initial_turn_id( other.m_max_initial_turn_id ),
        m_default_elem_by_elem_rolling( other.m_default_elem_by_elem_rolling ),
        m_has_beam_monitor_output( other.m_has_beam_monitor_output ),
        m_has_elem_by_elem_output( other.m_has_elem_by_elem_output )
    {
        using elem_by_elem_config_t = TrackJobBase::elem_by_elem_config_t;

        if( other.ownsOutputBuffer() )
        {

        }

        if( other.m_my_elem_by_elem_config.get() != nullptr )
        {
            this->m_my_elem_by_elem_config.reset(
                new elem_by_elem_config_t );

            *this->m_my_elem_by_elem_config =  *other.ptrElemByElemConfig();
        }
    }

    SIXTRL_HOST_FN TrackJobBase::TrackJobBase(
        TrackJobBase&& o ) SIXTRL_NOEXCEPT :
        m_type_str( std::move( o.m_type_str ) ),
        m_device_id_str( std::move( o.m_type_str ) ),
        m_config_str( std::move( o.m_config_str ) ),
        m_particle_set_indices( std::move( o.m_particle_set_indices ) ),
        m_beam_monitor_indices( std::move( o.m_beam_monitor_indices ) ),
        m_my_output_buffer( std::move( o.m_my_output_buffer ) ),
        m_my_elem_by_elem_config( std::move( o.m_my_elem_by_elem_config ) ),
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
        m_dump_elem_by_elem_turns( std::move( o.m_dump_elem_by_elem_turns ) ),
        m_type_id( std::move( o.m_type_id ) ),
        m_default_elem_by_elem_order( std::move(
            o.m_default_elem_by_elem_order ) ),
        m_min_particle_id( std::move( o.m_min_particle_id ) ),
        m_max_particle_id( std::move( o.m_max_particle_id ) ),
        m_min_element_id( std::move( o.m_min_element_id ) ),
        m_max_element_id( std::move( o.m_max_element_id ) ),
        m_min_initial_turn_id( std::move( o.m_min_initial_turn_id ) ),
        m_max_initial_turn_id( std::move( o.m_max_initial_turn_id ) ),
        m_default_elem_by_elem_rolling( std::move(
            o.m_default_elem_by_elem_rolling ) ),
        m_has_beam_monitor_output( std::move( o.m_has_beam_monitor_output ) ),
        m_has_elem_by_elem_output( std::move( o.m_has_elem_by_elem_output ) )
    {
        o.m_type_str.clear();
        o.m_device_id_str.clear();
        o.m_config_str.clear();

        o.doClearBaseImpl();
    }

    SIXTRL_HOST_FN TrackJobBase& TrackJobBase::operator=(
        TrackJobBase const& rhs )
    {
        if( this != &rhs )
        {
            this->m_type_str                    = rhs.m_type_str;
            this->m_device_id_str               = rhs.m_type_str;
            this->m_config_str                  = rhs.m_config_str;
            this->m_particle_set_indices        = rhs.m_particle_set_indices;
            this->m_beam_monitor_indices        = rhs.m_beam_monitor_indices;
            this->m_ptr_particles_buffer        = rhs.m_ptr_particles_buffer;
            this->m_ptr_beam_elem_buffer        = rhs.m_ptr_beam_elem_buffer;
            this->m_ptr_c_particles_buffer      = rhs.m_ptr_c_particles_buffer;
            this->m_ptr_c_beam_elem_buffer      = rhs.m_ptr_c_beam_elem_buffer;

            this->m_be_mon_output_buffer_offset =
                rhs.m_be_mon_output_buffer_offset;

            this->m_dump_elem_by_elem_turns    = rhs.m_dump_elem_by_elem_turns;
            this->m_type_id                    = rhs.m_type_id;

            this->m_elem_by_elem_output_offset =
                rhs.m_elem_by_elem_output_offset;

            this->m_default_elem_by_elem_order =
                rhs.m_default_elem_by_elem_order;

            this->m_min_particle_id            = rhs.m_min_particle_id;
            this->m_max_particle_id            = rhs.m_max_particle_id;
            this->m_min_element_id             = rhs.m_min_element_id;
            this->m_max_element_id             = rhs.m_max_element_id;
            this->m_min_initial_turn_id        = rhs.m_min_initial_turn_id;
            this->m_max_initial_turn_id        = rhs.m_max_initial_turn_id;

            this->m_default_elem_by_elem_rolling =
                rhs.m_default_elem_by_elem_rolling;

            this->m_has_beam_monitor_output = rhs.m_has_beam_monitor_output;
            this->m_has_elem_by_elem_output = rhs.m_has_elem_by_elem_output;

            if( rhs.ownsOutputBuffer() )
            {
               this->m_ptr_output_buffer    = nullptr;
               this->m_ptr_c_output_buffer  = nullptr;
            }
            else
            {
                this->m_ptr_output_buffer   = rhs.m_ptr_output_buffer;
                this->m_ptr_c_output_buffer = rhs.m_ptr_c_output_buffer;
            }

            if( rhs.m_my_elem_by_elem_config.get() != nullptr )
            {
                this->m_my_elem_by_elem_config.reset(
                    new elem_by_elem_config_t );

                *this->m_my_elem_by_elem_config =  *rhs.ptrElemByElemConfig();
            }
            else
            {
                this->m_my_elem_by_elem_config.reset( nullptr );
            }
        }

        return *this;
    }

    SIXTRL_HOST_FN TrackJobBase& TrackJobBase::operator=(
        TrackJobBase&& rhs ) SIXTRL_NOEXCEPT
    {
        if( this != &rhs )
        {
            this->m_type_str         = std::move( rhs.m_type_str );
            this->m_device_id_str    = std::move( rhs.m_type_str );
            this->m_config_str       = std::move( rhs.m_config_str );

            this->m_particle_set_indices = std::move( rhs.m_particle_set_indices );
            this->m_beam_monitor_indices = std::move( rhs.m_beam_monitor_indices );

            this->m_my_output_buffer = std::move( rhs.m_my_output_buffer );

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

            this->m_ptr_c_output_buffer =
                std::move( rhs.m_ptr_c_output_buffer );

            this->m_be_mon_output_buffer_offset =
                std::move( rhs.m_be_mon_output_buffer_offset );

            this->m_elem_by_elem_output_offset =
                std::move( rhs.m_elem_by_elem_output_offset );

            this->m_dump_elem_by_elem_turns =
                std::move( rhs.m_dump_elem_by_elem_turns );

            this->m_type_id = std::move( rhs.m_type_id );

            this->m_default_elem_by_elem_order =
                std::move( rhs.m_default_elem_by_elem_order );

            this->m_min_particle_id     = std::move( rhs.m_min_particle_id );
            this->m_max_particle_id     = std::move( rhs.m_max_particle_id );
            this->m_min_element_id      = std::move( rhs.m_min_element_id  );
            this->m_max_element_id      = std::move( rhs.m_max_element_id  );
            this->m_min_initial_turn_id = std::move( rhs.m_min_initial_turn_id);
            this->m_max_initial_turn_id = std::move( rhs.m_max_initial_turn_id);

            this->m_default_elem_by_elem_rolling =
                std::move( rhs.m_default_elem_by_elem_rolling );

            this->m_has_beam_monitor_output =
                std::move( rhs.m_has_beam_monitor_output );

            this->m_has_elem_by_elem_output =
                std::move( rhs.m_has_elem_by_elem_output );

            rhs.m_type_str.clear();
            rhs.m_device_id_str.clear();
            rhs.m_config_str.clear();

            rhs.m_type_id = type_t{ 0 };
            rhs.doClearBaseImpl();
        }

        return *this;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN void TrackJobBase::doClear()
    {
        this->doClearBaseImpl();
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doCollect()
    {
        return;
    }

    SIXTRL_HOST_FN TrackJobBase::track_status_t
    TrackJobBase::doTrackUntilTurn( size_type const )
    {
        return TrackJobBase::track_status_t{ -1 };
    }

    SIXTRL_HOST_FN TrackJobBase::track_status_t
    TrackJobBase::doTrackElemByElem( size_type const )
    {
        return TrackJobBase::track_status_t{ -1 };
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN bool TrackJobBase::doPrepareParticlesStructures(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT pb )
    {
        bool success = false;

        using size_t    = TrackJobBase::size_type;
        using p_index_t = TrackJobBase::particle_index_t;
        using p_limit_t = std::numeric_limits< p_index_t >;

        SIXTRL_STATIC_VAR size_t const ZERO = size_t{ 0 };
        SIXTRL_STATIC_VAR size_t const ONE  = size_t{ 1 };

        size_t const nn = this->numParticleSets();
        size_t const num_psets = ::NS(Buffer_get_num_of_objects)( pb );

        if( ( pb != nullptr ) && ( ( nn > ZERO ) || ( nn == ZERO ) ) &&
            ( !::NS(Buffer_needs_remapping)( pb ) ) && ( num_psets >= nn ) )
        {
            int ret = int{ -1 };

            p_index_t min_particle_id = p_limit_t::min();
            p_index_t max_particle_id = p_limit_t::max();

            p_index_t min_element_id  = p_limit_t::min();
            p_index_t max_element_id  = p_limit_t::max();

            p_index_t min_turn_id     = p_limit_t::min();
            p_index_t max_turn_id     = p_limit_t::max();

            size_t const first_index = ( nn > ZERO )
                ? this->particleSetIndex( ZERO ) : ZERO;

            if( ( nn <= ONE ) && ( first_index == ZERO ) )
            {
                ret = ::NS(Particles_get_min_max_attributes)(
                        ::NS(Particles_buffer_get_const_particles)( pb, ZERO ),
                        &min_particle_id, &max_particle_id, &min_element_id,
                        &max_element_id, &min_turn_id, &max_turn_id );
            }
            else if( nn > ZERO )
            {
                ret =
                ::NS(Particles_buffer_get_min_max_attributes_of_particles_set)(
                    pb, nn, this->particleSetIndicesBegin(),
                    &min_particle_id, &max_particle_id, &min_element_id,
                    &max_element_id, &min_turn_id, &max_turn_id );
            }

            if( ret == int{ 0 } )
            {
                this->doSetMinParticleId( min_particle_id );
                this->doSetMaxParticleId( max_particle_id );

                this->doSetMinElementId( min_element_id );
                this->doSetMaxElementId( max_element_id );

                this->doSetMinInitialTurnId( min_turn_id );
                this->doSetMaxInitialTurnId( max_turn_id );

                success = true;
            }
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobBase::doPrepareBeamElementsStructures(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT be_buffer )
    {
        bool success = false;

        using size_t      = TrackJobBase::size_type;
        using p_index_t   = TrackJobBase::particle_index_t;
        using buf_size_t  = ::NS(buffer_size_t);
        using obj_type_id = ::NS(object_type_id_t);
        using obj_it      = SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object) const*;

        SIXTRL_STATIC_VAR size_t const ZERO = size_t{ 0 };

        if( ( be_buffer != nullptr ) &&
            ( !::NS(Buffer_needs_remapping)( be_buffer ) ) &&
            ( ::NS(Buffer_get_num_of_objects)( be_buffer ) > ZERO ) &&
            ( ::NS(BeamElements_is_beam_elements_buffer)( be_buffer ) ) )
        {
            p_index_t min_element_id = p_index_t{ 0 };
            p_index_t max_element_id = std::numeric_limits< p_index_t >::max();

            buf_size_t num_elem_by_elem_objects = buf_size_t{ 0 };

            int ret = ::NS(ElemByElemConfig_get_min_max_element_id_from_buffer)(
                be_buffer, &min_element_id, &max_element_id,
                    &num_elem_by_elem_objects, p_index_t{ 0 } );

            if( ( ret == 0 ) &&
                ( num_elem_by_elem_objects > buf_size_t{ 0 } ) )
            {
                if( this->minElementId() > min_element_id )
                {
                    this->doSetMinElementId( min_element_id );
                }

                if( this->maxElementId() < max_element_id )
                {
                    this->doSetMaxElementId( max_element_id );
                }

                std::vector< size_t > be_monitor_indices(
                    ::NS(Buffer_get_num_of_objects)( be_buffer ), ZERO );

                be_monitor_indices.clear();

                size_t ii  = size_t{ 0 };
                obj_it it  = ::NS(Buffer_get_const_objects_begin)( be_buffer );
                obj_it end = ::NS(Buffer_get_const_objects_end)( be_buffer );

                for( ; it != end ; ++it, ++ii )
                {
                    obj_type_id const type_id = ::NS(Object_get_type_id)( it );

                    if( type_id == ::NS(OBJECT_TYPE_BEAM_MONITOR) )
                    {
                        be_monitor_indices.push_back( ii );
                    }
                }

                if( !be_monitor_indices.empty() )
                {
                    this->doSetBeamMonitorIndices(
                        be_monitor_indices.begin(),
                        be_monitor_indices.end() );
                }

                SIXTRL_ASSERT( be_monitor_indices.size() ==
                    this->numBeamMonitors() );
            }
            else if( ret == 0 )
            {
                size_t const dummy[] = { 0 };
                this->doSetBeamMonitorIndices( &dummy[ 0 ], &dummy[ 0 ] );

                SIXTRL_ASSERT( this->numBeamMonitors() == ZERO );
            }

            success = ( ret == 0 );
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobBase::doPrepareOutputStructures(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobBase::size_type const dump_elem_by_elem_turns )
    {
        bool success = false;

        using size_t                    = TrackJobBase::size_type;
        using buffer_t                  = TrackJobBase::buffer_t;
        using c_buffer_t                = TrackJobBase::c_buffer_t;
        using buf_size_t                = ::NS(buffer_size_t);
        using elem_by_elem_config_t     = TrackJobBase::elem_by_elem_config_t;
        using ptr_output_buffer_t       = TrackJobBase::ptr_output_buffer_t;
        using par_index_t               = TrackJobBase::particle_index_t;

        using ptr_elem_by_elem_config_t =
            TrackJobBase::ptr_elem_by_elem_config_t;

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
                return success;
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
            par_index_t max_elem_by_elem_turn_id    =
                this->minInitialTurnId() + dump_elem_by_elem_turns;

            int ret = ::NS(OutputBuffer_prepare_detailed)(
                beam_elements_buffer, output_buffer,
                this->minParticleId(), this->maxParticleId(),
                this->minElementId(),  this->maxElementId(),
                this->minInitialTurnId(), this->maxInitialTurnId(),
                dump_elem_by_elem_turns,
                &elem_by_elem_out_idx_offset, &be_monitor_out_idx_offset,
                &max_elem_by_elem_turn_id );

            if( ( ret == 0 ) && ( dump_elem_by_elem_turns > ZERO ) )
            {
                ptr_elem_by_elem_config_t conf( new elem_by_elem_config_t );
                ::NS(ElemByElemConfig_preset)( conf.get() );

                ret = ::NS(ElemByElemConfig_init_detailed)(
                    conf.get(), this->defaultElemByElemOrder(),
                    this->minParticleId(), this->maxParticleId(),
                    this->minElementId(),  this->maxElementId(),
                    this->minInitialTurnId(), max_elem_by_elem_turn_id,
                    this->defaultElemByElemRolling() );

                if( ret == 0 )
                {
                    this->doUpdateStoredElemByElemConfig( std::move( conf ) );
                    this->doSetNumElemByElemOutputTurns(
                        dump_elem_by_elem_turns );

                    SIXTRL_ASSERT( this->hasElemByElemConfig() );
                }
            }
            else if( ret == 0 )
            {
                ptr_elem_by_elem_config_t dummy( nullptr );
                this->doUpdateStoredElemByElemConfig( std::move( dummy ) );
                this->doSetNumElemByElemOutputTurns( ZERO );

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

    bool TrackJobBase::doAssignOutputBufferToBeamMonitors(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer )
    {
        using size_t   = TrackJobBase::size_type;
        using config_t = TrackJobBase::elem_by_elem_config_t;

        size_t const num_be_monitors  = this->numBeamMonitors();
        size_t const be_mon_offset    = this->beamMonitorsOutputBufferOffset();
        size_t const elem_elem_offset = this->elemByElemOutputBufferOffset();
        config_t* elem_conf = this->ptrElemByElemConfig();

        size_t const num_out_objects  =
            ::NS(Buffer_get_num_of_objects)( output_buffer );

        bool success = false;

        if( ( output_buffer != nullptr ) && ( beam_elem_buffer != nullptr ) )
        {
            this->doSetBeamMonitorOutputEnabledFlag( false );
            this->doSetElemByElemOutputEnabledFlag( false );

            success = true;

            if( elem_conf != nullptr )
            {
                success = false;

                if( ( elem_elem_offset < num_out_objects ) &&
                    ( ( ( num_be_monitors == size_t{ 0 } ) &&
                        ( be_mon_offset >= elem_elem_offset ) ) ||
                      ( ( num_be_monitors > size_t{ 0 } ) &&
                        ( be_mon_offset >  elem_elem_offset ) ) ) &&
                    ( 0 == ::NS(ElemByElemConfig_assign_output_buffer)(
                        elem_conf, output_buffer, elem_elem_offset ) ) )
                {
                    this->doSetElemByElemOutputEnabledFlag( true );
                    success = true;
                }
            }

            if( ( success ) && ( num_be_monitors > size_t{ 0 } ) )
            {
                success = false;

                if( ( be_mon_offset < num_out_objects ) &&
                    ( ( ( elem_conf == nullptr ) &&
                        ( elem_elem_offset <= be_mon_offset ) ) ||
                      ( ( elem_conf != nullptr ) &&
                        ( elem_elem_offset < be_mon_offset ) ) ) &&
                    ( 0 == ::NS(BeamMonitor_assign_output_buffer_from_offset)(
                        beam_elem_buffer, output_buffer,
                        this->minInitialTurnId(), be_mon_offset ) ) )
                {
                    this->doSetBeamMonitorOutputEnabledFlag( true );
                    success = true;
                }
            }
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobBase::doReset(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT output_buffer,
        TrackJobBase::size_type const dump_elem_by_elem_turns )
    {
        bool success = false;

        using flag_t = SIXTRL_CXX_NAMESPACE::track_job_output_flag_t;

        if( ( this->doPrepareParticlesStructures( particles_buffer ) ) &&
            ( this->doPrepareBeamElementsStructures( beam_elem_buffer ) ) )
        {
            success = true;

            flag_t const needs_output = ::NS(TrackJob_needs_output_buffer)(
                particles_buffer, beam_elem_buffer, dump_elem_by_elem_turns );

            if( needs_output != ::NS(TRACK_JOB_OUTPUT_NONE) )
            {
                success = this->doPrepareOutputStructures( particles_buffer,
                    beam_elem_buffer, output_buffer, dump_elem_by_elem_turns );
            }

            if( ( success ) && ( this->hasOutputBuffer() ) &&
                ( needs_output != ::NS(TRACK_JOB_OUTPUT_NONE ) ) )
            {
                success = this->doAssignOutputBufferToBeamMonitors(
                    beam_elem_buffer, this->ptrCOutputBuffer() );
            }

            if( ( success ) && ( output_buffer != nullptr ) &&
                ( !this->ownsOutputBuffer() ) )
            {
                this->doSetPtrCOutputBuffer( output_buffer );
            }
        }

        return success;
    }

    SIXTRL_HOST_FN bool TrackJobBase::doAssignNewOutputBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer )
    {
        bool success = false;

        track_job_output_flag_t const needs_output =
        ::NS(TrackJob_needs_output_buffer)( this->ptrCParticlesBuffer(),
            this->ptrCBeamElementsBuffer(), this->numElemByElemTurns() );

        if( needs_output != ::NS(TRACK_JOB_OUTPUT_NONE) )
        {
            success = this->doPrepareOutputStructures(
                this->ptrCParticlesBuffer(), this->ptrCBeamElementsBuffer(),
                ptr_output_buffer, this->numElemByElemTurns() );
        }

        if( ( success ) &&
            ( ( needs_output & ::NS(TRACK_JOB_OUTPUT_BEAM_MONITORS) ) ==
                ::NS(TRACK_JOB_OUTPUT_BEAM_MONITORS) ) &&
            ( this->hasOutputBuffer() ) && ( this->hasBeamMonitorOutput() ) )
        {
            success = this->doAssignOutputBufferToBeamMonitors(
                this->ptrCBeamElementsBuffer(), this->ptrCOutputBuffer() );
        }

        return success;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_HOST_FN void TrackJobBase::doParseConfigStr(
        const char *const SIXTRL_RESTRICT config_str )
    {
        this->doParseConfigStrBaseImpl( config_str );
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetDeviceIdStr(
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

        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetConfigStr(
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

        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetPtrParticleBuffer(
        TrackJobBase::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetPtrCParticleBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ( this->m_ptr_particles_buffer   != nullptr ) &&
            ( this->m_ptr_c_particles_buffer ==
              this->m_ptr_particles_buffer->getCApiPtr() ) &&
            ( this->m_ptr_particles_buffer->getCApiPtr() != ptr_buffer ) )
        {
            this->m_ptr_particles_buffer = nullptr;
        }

        this->m_ptr_c_particles_buffer = ptr_buffer;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetPtrBeamElementsBuffer(
        TrackJobBase::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetPtrCBeamElementsBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ( this->m_ptr_beam_elem_buffer != nullptr ) &&
            ( this->m_ptr_c_beam_elem_buffer ==
              this->m_ptr_beam_elem_buffer->getCApiPtr() ) &&
            ( this->m_ptr_beam_elem_buffer->getCApiPtr() != ptr_buffer ) )
        {
            this->m_ptr_beam_elem_buffer = nullptr;
        }

        this->m_ptr_c_beam_elem_buffer = ptr_buffer;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetPtrOutputBuffer(
        TrackJobBase::buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
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
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetPtrCOutputBuffer(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        if( ( this->m_ptr_output_buffer != nullptr ) &&
            ( this->m_ptr_c_output_buffer ==
              this->m_ptr_output_buffer->getCApiPtr() ) &&
            ( this->m_ptr_output_buffer->getCApiPtr() != ptr_buffer ) )
        {
            this->m_ptr_output_buffer = nullptr;
        }

        this->m_ptr_c_output_buffer = ptr_buffer;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetBeamMonitorOutputBufferOffset(
        TrackJobBase::size_type const output_buffer_offset ) SIXTRL_NOEXCEPT
    {
        this->m_be_mon_output_buffer_offset = output_buffer_offset;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetNumElemByElemOutputTurns(
        TrackJobBase::size_type const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT
    {
        this->m_dump_elem_by_elem_turns = dump_elem_by_elem_turns;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetElemByElemOutputIndexOffset(
        TrackJobBase::size_type const elem_by_elem_output_offset ) SIXTRL_NOEXCEPT
    {
        this->m_elem_by_elem_output_offset = elem_by_elem_output_offset;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetBeamMonitorOutputEnabledFlag(
        bool const has_beam_monitor_output ) SIXTRL_NOEXCEPT
    {
        this->m_has_beam_monitor_output = has_beam_monitor_output;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetElemByElemOutputEnabledFlag(
        bool const elem_by_elem_flag ) SIXTRL_NOEXCEPT
    {
        this->m_has_elem_by_elem_output = elem_by_elem_flag;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doInitDefaultParticleSetIndices()
    {
        this->m_particle_set_indices.clear();
        this->m_particle_set_indices.push_back( TrackJobBase::size_type{ 0 } );

        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doInitDefaultBeamMonitorIndices()
    {
        this->m_beam_monitor_indices.clear();
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetMinParticleId(
        TrackJobBase::particle_index_t const min_particle_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_particle_id = min_particle_id;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetMaxParticleId(
        TrackJobBase::particle_index_t const max_particle_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_particle_id = max_particle_id;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetMinElementId(
        TrackJobBase::particle_index_t const min_element_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_element_id = min_element_id;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetMaxElementId(
        TrackJobBase::particle_index_t const max_element_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_element_id = max_element_id;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetMinInitialTurnId(
        TrackJobBase::particle_index_t const min_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_min_initial_turn_id = min_turn_id;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doSetMaxInitialTurnId(
        TrackJobBase::particle_index_t const max_turn_id ) SIXTRL_NOEXCEPT
    {
        this->m_max_initial_turn_id = max_turn_id;
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doUpdateStoredOutputBuffer(
        TrackJobBase::ptr_output_buffer_t&& ptr_output_buffer ) SIXTRL_NOEXCEPT
    {
        this->doSetPtrOutputBuffer( ptr_output_buffer.get() );
        this->m_my_output_buffer = std::move( ptr_output_buffer );
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doUpdateStoredElemByElemConfig(
        TrackJobBase::ptr_elem_by_elem_config_t&& ptr_config ) SIXTRL_NOEXCEPT
    {
        this->m_my_elem_by_elem_config = std::move( ptr_config );
        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doClearBaseImpl() SIXTRL_NOEXCEPT
    {
        using p_index_t = TrackJobBase::particle_index_t;

        this->doInitDefaultParticleSetIndices();
        this->doInitDefaultBeamMonitorIndices();

        this->m_my_output_buffer.reset( nullptr );
        this->m_my_elem_by_elem_config.reset( nullptr );

        this->m_ptr_particles_buffer         = nullptr;
        this->m_ptr_beam_elem_buffer         = nullptr;
        this->m_ptr_output_buffer            = nullptr;

        this->m_ptr_c_particles_buffer       = nullptr;
        this->m_ptr_c_beam_elem_buffer       = nullptr;
        this->m_ptr_c_output_buffer          = nullptr;

        this->m_be_mon_output_buffer_offset  = TrackJobBase::size_type{ 0 };
        this->m_elem_by_elem_output_offset   = TrackJobBase::size_type{ 0 };
        this->m_dump_elem_by_elem_turns      = TrackJobBase::size_type{ 0 };
        this->m_default_elem_by_elem_order   = ::NS(ELEM_BY_ELEM_ORDER_DEFAULT);

        this->m_min_particle_id              = p_index_t{ 0 };
        this->m_max_particle_id              = p_index_t{ 0 };

        this->m_min_element_id               = p_index_t{ 0 };
        this->m_max_element_id               = p_index_t{ 0 };

        this->m_min_initial_turn_id          = p_index_t{ 0 };
        this->m_max_initial_turn_id          = p_index_t{ 0 };

        this->m_default_elem_by_elem_rolling = true;
        this->m_has_beam_monitor_output      = false;
        this->m_has_elem_by_elem_output      = false;

        return;
    }

    SIXTRL_HOST_FN void TrackJobBase::doParseConfigStrBaseImpl(
        const char *const SIXTRL_RESTRICT config_str )
    {
        ( void )config_str;
        return;
    }

    /* ===================================================================== */

    SIXTRL_HOST_FN track_job_output_flag_t TrackJob_needs_output_buffer(
        Buffer const& SIXTRL_RESTRICT_REF particles_buffer,
        Buffer const& SIXTRL_RESTRICT_REF beam_elem_buffer,
        Buffer::size_type const dump_elem_by_elem_turns ) SIXTRL_NOEXCEPT
    {
        return ::NS(TrackJob_needs_output_buffer)(
            particles_buffer.getCApiPtr(), beam_elem_buffer.getCApiPtr(),
                dump_elem_by_elem_turns );
    }
}

#endif /* defined( __cplusplus ) */


SIXTRL_HOST_FN void NS(TrackJob_clear)( NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    if( job != nullptr ) job->clear();
}

SIXTRL_HOST_FN void NS(TrackJob_collect)( NS(TrackJobBase)* SIXTRL_RESTRICT j )
{
    if( j != nullptr ) j->collect();
}

SIXTRL_HOST_FN bool NS(TrackJob_reset)( NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elem_buffer, ptr_output_buffer )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_reset_with_output)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr )
        ? job->reset( particles_buffer, beam_elem_buffer, output_buffer,
                      dump_elem_by_elem_turns )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_reset_detailed)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT particles_buffer,
    NS(buffer_size_t) const num_particle_sets,
    NS(buffer_size_t) const* SIXTRL_RESTRICT particle_set_indices_begin,
    NS(Buffer)* SIXTRL_RESTRICT beam_elem_buffer,
    NS(Buffer)* SIXTRL_RESTRICT output_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    return ( job != nullptr ) ? job->reset( particles_buffer, num_particle_sets,
            particle_set_indices_begin, beam_elem_buffer, output_buffer,
                dump_elem_by_elem_turns )
        : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_assign_output_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(Buffer)* SIXTRL_RESTRICT ptr_output_buffer )
{
    return ( job != nullptr )
        ? job->assignOutputBuffer( ptr_output_buffer ) : false;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(track_job_type_t) NS(TrackJob_get_type_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->type() : ::NS(track_job_type_t){ -1 };
}

SIXTRL_HOST_FN char const* NS(TrackJob_get_type_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->typeStr().c_str() : nullptr;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_device_id_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasDeviceIdStr() : false;
}

SIXTRL_HOST_FN char const* NS(TrackJob_get_device_id_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->deviceIdStr().c_str() : nullptr;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_config_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasConfigStr() : false;
}

SIXTRL_HOST_FN char const* NS(TrackJob_get_config_str)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->configStr().c_str() : nullptr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_num_particle_sets)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numParticleSets() : ::NS(buffer_size_t){ 0 };
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_particle_set_indices_begin)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->particleSetIndicesBegin() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_particle_set_indices_end)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->particleSetIndicesEnd() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_particle_set_index)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->particleSetIndex( n );
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_min_particle_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minParticleId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_max_particle_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxParticleId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_min_element_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minElementId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_max_element_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxElementId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_min_initial_turn_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->minInitialTurnId() : index_t{ -1 };
}

SIXTRL_HOST_FN NS(particle_index_t) NS(TrackJob_get_max_initial_turn_id)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    using index_t = ::NS(particle_index_t);
    return ( job != nullptr ) ? job->maxInitialTurnId() : index_t{ -1 };
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJob_get_particles_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCParticlesBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer) const* NS(TrackJob_get_const_particles_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCParticlesBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer)*
NS(TrackJob_get_beam_elements_buffer)( NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCBeamElementsBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJob_get_const_beam_elements_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCBeamElementsBuffer() : nullptr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(TrackJob_has_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasOutputBuffer() : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_owns_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ownsOutputBuffer() : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_elem_by_elem_output)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasElemByElemOutput() : false;
}

SIXTRL_HOST_FN bool NS(TrackJob_has_beam_monitor_output)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasBeamMonitorOutput() : false;
}

SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_beam_monitor_output_buffer_offset)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->beamMonitorsOutputBufferOffset();
}

SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_elem_by_elem_output_buffer_offset)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    SIXTRL_ASSERT( job != nullptr );
    return job->elemByElemOutputBufferOffset();
}

SIXTRL_HOST_FN NS(buffer_size_t)
NS(TrackJob_get_num_elem_by_elem_turns)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numElemByElemTurns() : ::NS(buffer_size_t){ 0 };
}

SIXTRL_HOST_FN NS(Buffer)* NS(TrackJob_get_output_buffer)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCOutputBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer) const*
NS(TrackJob_get_const_output_buffer)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrCOutputBuffer() : nullptr;
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(TrackJob_has_beam_monitors)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasBeamMonitors() : false;
}

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_num_beam_monitors)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->numBeamMonitors() : ::NS(buffer_size_t){ 0 };
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_beam_monitor_indices_begin)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->beamMonitorIndicesBegin() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) const*
NS(TrackJob_get_beam_monitor_indices_end)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->beamMonitorIndicesEnd() : nullptr;
}

SIXTRL_HOST_FN NS(buffer_size_t) NS(TrackJob_get_beam_monitor_index)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job,
    NS(buffer_size_t) const n )
{
    return ( job != nullptr )
        ? job->beamMonitorIndex( n ) : ::NS(buffer_size_t){ 0 };
}

/* ------------------------------------------------------------------------- */

SIXTRL_HOST_FN bool NS(TrackJob_has_elem_by_elem_config)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->hasElemByElemConfig() : false;
}

SIXTRL_HOST_FN NS(ElemByElemConfig) const*
NS(TrackJob_get_elem_by_elem_config)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->ptrElemByElemConfig() : nullptr;
}

SIXTRL_HOST_FN bool NS(TrackJob_is_elem_by_elem_config_rolling)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->elemByElemRolling() : false;
}

SIXTRL_HOST_FN bool
NS(TrackJob_get_default_elem_by_elem_config_rolling_flag)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr ) ? job->defaultElemByElemRolling() : false;
}

SIXTRL_HOST_FN void
NS(TrackJob_set_default_elem_by_elem_config_rolling_flag)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job, bool const is_rolling_flag )
{
    if( job != nullptr )
    {
        job->setDefaultElemByElemRolling( is_rolling_flag );
    }

    return;
}

SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJob_get_elem_by_elem_config_order)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->elemByElemOrder() : ::NS(ELEM_BY_ELEM_ORDER_INVALID);
}

SIXTRL_HOST_FN NS(elem_by_elem_order_t)
NS(TrackJob_get_default_elem_by_elem_config_order)(
    const NS(TrackJobBase) *const SIXTRL_RESTRICT job )
{
    return ( job != nullptr )
        ? job->defaultElemByElemOrder()
        : ::NS(ELEM_BY_ELEM_ORDER_INVALID);
}

SIXTRL_HOST_FN void
NS(TrackJob_set_default_elem_by_elem_config_order)(
    NS(TrackJobBase)* SIXTRL_RESTRICT job,
    NS(elem_by_elem_order_t) const order )
{
    if( job != nullptr )
    {
        job->setDefaultElemByElemOrder( order );
    }

    return;
}

SIXTRL_HOST_FN ::NS(track_job_output_flag_t) NS(TrackJob_needs_output_buffer)(
    const NS(Buffer) *const SIXTRL_RESTRICT particles_buffer,
    const NS(Buffer) *const SIXTRL_RESTRICT belem_buffer,
    NS(buffer_size_t) const dump_elem_by_elem_turns )
{
    using size_t          = ::NS(buffer_size_t);
    using obj_iter_t      = ::NS(Object) const*;
    using type_id_t       = ::NS(object_type_id_t);
    using address_t       = ::NS(buffer_addr_t);

    ::NS(track_job_output_flag_t) flags = ::NS(TRACK_JOB_OUTPUT_NONE);

    obj_iter_t be_it = ::NS(Buffer_get_const_objects_begin)( belem_buffer );
    size_t const num_belems = ::NS(Buffer_get_num_of_objects)( belem_buffer );

    SIXTRL_ASSERT( ::NS(BeamElements_is_beam_elements_buffer)(
        belem_buffer ) );

    SIXTRL_ASSERT( ::NS(Buffer_is_particles_buffer)( particles_buffer ) );
    SIXTRL_ASSERT( ::NS(Particles_buffer_get_total_num_of_particles)(
        particles_buffer ) > ::NS(particle_num_elements_t){ 0 } );

    if( dump_elem_by_elem_turns > size_t{ 0 } )
    {
        flags |= ::NS(TRACK_JOB_OUTPUT_ELEM_BY_ELEM);
    }

    if( ( be_it != nullptr ) && ( num_belems > size_t{ 0 } ) );
    {
        obj_iter_t be_end = be_it;
        std::advance( be_end, std::ptrdiff_t{ -1 } );
        std::advance( be_it,  num_belems - size_t{ 1 } );

        for( ; be_it != be_end ; --be_it )
        {
            address_t const addr   = ::NS(Object_get_begin_addr)( be_it );
            type_id_t const type   = ::NS(Object_get_type_id)( be_it );
            size_t const  obj_size = ::NS(Object_get_size)( be_it );

            if( ( addr != address_t{ 0 } ) &&
                ( type == ::NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
                ( obj_size >= sizeof( ::NS(BeamMonitor) ) ) )
            {
                ::NS(BeamMonitor) const* be_monitor =
                    static_cast< ::NS(BeamMonitor) const* >(
                        reinterpret_cast< void const* >(
                            static_cast< uintptr_t >( addr ) ) );

                if( ( be_monitor != nullptr ) &&
                    ( ::NS(BeamMonitor_get_num_stores)( be_monitor ) > 0 ) )
                {
                    flags |= ::NS(TRACK_JOB_OUTPUT_BEAM_MONITORS);
                    break;
                }
            }
        }
    }

    return flags;
}

/* end: sixtracklib/common/internal/track_job_base.cpp */
