#ifndef SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_CXX_HEADER_HPP__
#define SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_CXX_HEADER_HPP__

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <type_traits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/be_monitor/be_monitor.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    struct BeamMonitor : public ::NS(BeamMonitor)
    {
        using index_t   = ::NS(be_monitor_index_t);
        using turn_t    = ::NS(be_monitor_turn_t);
        using address_t = ::NS(be_monitor_addr_t);
        using flag_t    = ::NS(be_monitor_flag_t);
        using type_id_t = ::NS(object_type_id_t);
        using c_api_t   = ::NS(BeamMonitor);
        using size_type = ::NS(buffer_size_t);
        using buffer_t  = ::NS(Buffer);

        SIXTRL_FN BeamMonitor() = default;
        SIXTRL_FN BeamMonitor( BeamMonitor const& other ) = default;
        SIXTRL_FN BeamMonitor( BeamMonitor&& other ) = default;

        SIXTRL_FN BeamMonitor& operator=( BeamMonitor const& rhs ) = default;
        SIXTRL_FN BeamMonitor& operator=( BeamMonitor&& rhs ) = default;

        SIXTRL_FN ~BeamMonitor() = default;

        /* ----------------------------------------------------------------- */

        #if !defined( _GPUCODE )

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        BeamMonitor const* FromBuffer(
            Buffer const& SIXTRL_RESTRICT_REF buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC
        BeamMonitor const* FromBuffer(
            const ::NS(Buffer) *const SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC BeamMonitor* FromBuffer(
            Buffer& SIXTRL_RESTRICT_REF buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC BeamMonitor* FromBuffer(
            ::NS(Buffer)* SIXTRL_RESTRICT buffer,
            size_type const be_index ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC BeamMonitor const*
        FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const NS(Object) *const
                SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC BeamMonitor*
        FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object)*
                SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_STATIC bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC BeamMonitor*
        CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC SIXTRL_BE_ARGPTR_DEC BeamMonitor* AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            turn_t const num_stores, turn_t const start,
            turn_t const skip, address_t const out_address,
            index_t const min_particle_id, index_t const max_particle_id,
            bool const is_rolling, bool const is_turn_ordered );

        SIXTRL_FN SIXTRL_STATIC bool arePresentInBuffer(
            buffer_t& SIXTRL_RESTRICT_REF belements_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC size_type getNumOfBeamMonitorObjects(
            buffer_t& SIXTRL_RESTRICT_REF belements_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_STATIC void clearAll(
            buffer_t& SIXTRL_RESTRICT_REF belements_buffer ) SIXTRL_NOEXCEPT;

        #endif /* !defined( _GPUCODE ) */


        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumSlots( size_type const slot_size =
            ::NS(BUFFER_DEFAULT_SLOT_SIZE) ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN turn_t    getNumStores()      const SIXTRL_NOEXCEPT;
        SIXTRL_FN turn_t    getStart()          const SIXTRL_NOEXCEPT;
        SIXTRL_FN turn_t    getSkip()           const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool      isRolling()         const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool      isTurnOrdered()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool      isParticleOrdered() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getOutAddress()     const SIXTRL_NOEXCEPT;
        SIXTRL_FN index_t   getMinParticleId()  const SIXTRL_NOEXCEPT;
        SIXTRL_FN index_t   getMaxParticleId()  const SIXTRL_NOEXCEPT;


        /* ----------------------------------------------------------------- */

        SIXTRL_FN void setNumStores( turn_t const num_stores ) SIXTRL_NOEXCEPT;
        SIXTRL_FN void setStart( turn_t const start_turn )     SIXTRL_NOEXCEPT;
        SIXTRL_FN void setSkip(  turn_t const skip_turns )     SIXTRL_NOEXCEPT;
        SIXTRL_FN void setIsRolling( bool const is_rolling )   SIXTRL_NOEXCEPT;

        SIXTRL_FN void setIsTurnOrdered(
            bool const is_turn_ordered ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setIsParticleOrdered(
            bool const is_particle_ordered ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setOutAddress(
            address_t const out_address ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setMinParticleId(
            index_t const min_particle_id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setMaxParticleId(
            index_t const max_particle_id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void clear() SIXTRL_NOEXCEPT;
    };

    template<> struct ObjectTypeTraits< ::NS(BeamMonitor) >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_BEAM_MONITOR);
        }
    };

    template<> struct ObjectTypeTraits< BeamMonitor >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_BEAM_MONITOR);
        }
    };

    #if !defined( _GPUCODE )

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_new(
        Buffer& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_new(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer );

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        BeamMonitor::turn_t    const num_stores,
        BeamMonitor::turn_t    const start,
        BeamMonitor::turn_t    const skip,
        BeamMonitor::address_t const out_address,
        BeamMonitor::index_t   const min_particle_id,
        BeamMonitor::index_t   const max_particle_id,
        bool const is_rolling, bool const is_turn_ordered );

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        BeamMonitor::turn_t    const num_stores,
        BeamMonitor::turn_t    const start,
        BeamMonitor::turn_t    const skip,
        BeamMonitor::address_t const out_address,
        BeamMonitor::index_t   const min_particle_id,
        BeamMonitor::index_t   const max_particle_id,
        bool const is_rolling, bool const is_turn_ordered );

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        BeamMonitor const& SIXTRL_RESTRICT_REF orig );

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        BeamMonitor const& SIXTRL_RESTRICT_REF orig );

    bool BeamMonitor_insert_end_of_turn_monitors(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        BeamMonitor::turn_t const turn_by_turn_start,
        BeamMonitor::turn_t const num_turn_by_turn_turns,
        BeamMonitor::turn_t const target_num_turns,
        BeamMonitor::turn_t const skip_turns,
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC Buffer::object_t* prev_node );

    bool BeamMonitor_insert_end_of_turn_monitors(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        BeamMonitor::turn_t const turn_by_turn_start,
        BeamMonitor::turn_t const num_turn_by_turn_turns,
        BeamMonitor::turn_t const target_num_turns,
        BeamMonitor::turn_t const skip_turns,
        Buffer::size_type const insert_at_index );

    bool BeamMonitor_insert_end_of_turn_monitors(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(be_monitor_turn_t) const turn_by_turn_start,
        ::NS(be_monitor_turn_t) const num_turn_by_turn_turns,
        ::NS(be_monitor_turn_t) const target_num_turns,
        ::NS(be_monitor_turn_t) const skip_turns,
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object)* prev_node );

    bool BeamMonitor_insert_end_of_turn_monitors(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(be_monitor_turn_t) const turn_by_turn_start,
        ::NS(be_monitor_turn_t) const num_turn_by_turn_turns,
        ::NS(be_monitor_turn_t) const target_num_turns,
        ::NS(be_monitor_turn_t) const skip_turns,
        ::NS(buffer_size_t) const insert_at_index );

    #endif /* !defined( _GPUCODE ) */
}

/* ************************************************************************* *
 * *** Implementation of inline and template member functions          ***** *
 * ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
     #if !defined( _GPUCODE )

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor const*
    BeamMonitor::FromBuffer( Buffer const& SIXTRL_RESTRICT_REF buffer,
        BeamMonitor::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        return BeamMonitor::FromBufferObject( buffer[ be_index ] );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor const*
    BeamMonitor::FromBuffer( const ::NS(Buffer) *const SIXTRL_RESTRICT buffer,
        BeamMonitor::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        return BeamMonitor::FromBufferObject(
            ::NS(Buffer_get_const_object)( buffer, be_index ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor::FromBuffer(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        BeamMonitor::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        return BeamMonitor::FromBufferObject( buffer[ be_index ] );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor::FromBuffer(
        ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        BeamMonitor::size_type const be_index ) SIXTRL_NOEXCEPT
    {
        return BeamMonitor::FromBufferObject(
            ::NS(Buffer_get_object)( buffer, be_index ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor const*
    BeamMonitor::FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
        NS(Object) *const SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using belement_t = BeamMonitor;
        using ptr_beam_elem_t = SIXTRL_BUFFER_OBJ_ARGPTR_DEC belement_t const*;

        if( ( be_info != nullptr ) &&
            ( ::NS(Object_get_type_id)( be_info ) ==
              ::NS(OBJECT_TYPE_BEAM_MONITOR) ) &&
            ( ::NS(Object_get_size)( be_info ) >= sizeof( BeamMonitor ) ) )
        {
            return reinterpret_cast< ptr_beam_elem_t >(
                static_cast< uintptr_t >( ::NS(Object_get_begin_addr)(
                    be_info ) ) );
        }

        return nullptr;
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor*
    BeamMonitor::FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object)*
        SIXTRL_RESTRICT be_info ) SIXTRL_NOEXCEPT
    {
        using  _this_t        = BeamMonitor;
        using beam_element_t  = _this_t;
        using ptr_beam_elem_t = SIXTRL_BUFFER_OBJ_DATAPTR_DEC beam_element_t*;

        using object_t        = ::NS(Object);
        using ptr_const_obj_t = SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t const*;

        ptr_const_obj_t const_be_info = be_info;

        return const_cast< ptr_beam_elem_t >(
            _this_t::FromBufferObject( const_be_info ) );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE bool BeamMonitor::CanAddToBuffer(
        BeamMonitor::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_ARGPTR_DEC BeamMonitor::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_BUFFER_ARGPTR_DEC BeamMonitor::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_BUFFER_ARGPTR_DEC BeamMonitor::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_can_be_added)( &buffer, ptr_requ_objects,
            ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor*
    BeamMonitor::CreateNewOnBuffer( BeamMonitor::buffer_t& SIXTRL_RESTRICT_REF
        buffer ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC BeamMonitor*;
        return static_cast< ptr_t >( ::NS(BeamMonitor_new)( &buffer ) );
    }

    SIXTRL_INLINE SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor::AddToBuffer(
        BeamMonitor::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BeamMonitor::turn_t const num_stores,
        BeamMonitor::turn_t const start,
        BeamMonitor::turn_t const skip,
        BeamMonitor::address_t const out_address,
        BeamMonitor::index_t const min_particle_id,
        BeamMonitor::index_t const max_particle_id,
        bool const is_rolling, bool const is_turn_ordered )
    {
        using ptr_t = SIXTRL_BE_ARGPTR_DEC BeamMonitor*;
        return static_cast< ptr_t >( ::NS(BeamMonitor_add)(
            &buffer, num_stores, start, skip, out_address, min_particle_id,
                max_particle_id, is_rolling, is_turn_ordered ) );
    }

    SIXTRL_INLINE bool BeamMonitor::arePresentInBuffer( BeamMonitor::buffer_t&
        SIXTRL_RESTRICT_REF belements_buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_are_present_in_buffer)( &belements_buffer );
    }

    SIXTRL_INLINE BeamMonitor::size_type
    BeamMonitor::getNumOfBeamMonitorObjects( BeamMonitor::buffer_t&
        SIXTRL_RESTRICT_REF belements_buffer ) SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_num_monitors_in_buffer)( &belements_buffer );
    }

    SIXTRL_INLINE void BeamMonitor::clearAll( BeamMonitor::buffer_t&
        SIXTRL_RESTRICT_REF belements_buffer ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_reset_all_in_buffer)( &belements_buffer );
    }

    #endif /* !defined( _GPUCODE ) */

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE BeamMonitor::type_id_t
    BeamMonitor::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_type_id)();
    }

    SIXTRL_INLINE BeamMonitor::size_type
    BeamMonitor::getNumDataPtrs() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_num_dataptrs)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BeamMonitor::size_type
    BeamMonitor::getNumSlots( size_type const slot_size ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_num_slots)( this->getCApiPtr(), slot_size );
    }

    SIXTRL_INLINE BeamMonitor::c_api_t const*
    BeamMonitor::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        using c_api_t = BeamMonitor::c_api_t;
        return reinterpret_cast< SIXTRL_BE_ARGPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE BeamMonitor::c_api_t*
    BeamMonitor::getCApiPtr() SIXTRL_NOEXCEPT
    {
        using c_api_t = BeamMonitor::c_api_t;
        return const_cast< SIXTRL_BE_ARGPTR_DEC c_api_t* >(
            static_cast< BeamMonitor const& >( *this ).getCApiPtr() );
    }

    SIXTRL_INLINE void BeamMonitor::preset() SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_preset)( this->getCApiPtr() );
        return;
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE BeamMonitor::turn_t
    BeamMonitor::getNumStores() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_num_stores)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BeamMonitor::turn_t
    BeamMonitor::getStart() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_start)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BeamMonitor::turn_t
    BeamMonitor::getSkip() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_skip)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool BeamMonitor::isRolling() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_is_rolling)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool BeamMonitor::isTurnOrdered() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_is_turn_ordered)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool BeamMonitor::isParticleOrdered() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_is_particle_ordered)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BeamMonitor::address_t
    BeamMonitor::getOutAddress() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_out_address)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BeamMonitor::index_t
    BeamMonitor::getMinParticleId() const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_min_particle_id)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BeamMonitor::index_t
    BeamMonitor::getMaxParticleId()  const SIXTRL_NOEXCEPT
    {
        return ::NS(BeamMonitor_max_particle_id)( this->getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE void BeamMonitor::setNumStores(
        BeamMonitor::turn_t const num_stores ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_num_stores)( this->getCApiPtr(), num_stores );
        return;
    }

    SIXTRL_INLINE void BeamMonitor::setStart(
        BeamMonitor::turn_t const start_turn ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_start)( this->getCApiPtr(), start_turn );
        return;
    }

    SIXTRL_INLINE void BeamMonitor::setSkip(
        BeamMonitor::turn_t const skip_turns ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_skip)( this->getCApiPtr(), skip_turns );
        return;
    }

    SIXTRL_INLINE void BeamMonitor::setIsRolling(
        bool const is_rolling ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_is_rolling)( this->getCApiPtr(), is_rolling );
        return;
    }

    SIXTRL_INLINE void BeamMonitor::setIsTurnOrdered(
        bool const is_turn_ordered ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_is_turn_ordered)(
            this->getCApiPtr(), is_turn_ordered );

        return;
    }

    SIXTRL_INLINE void BeamMonitor::setIsParticleOrdered(
        bool const is_particle_ordered ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_is_particle_ordered)(
            this->getCApiPtr(), is_particle_ordered );

        return;
    }

    SIXTRL_INLINE void BeamMonitor::setOutAddress(
        BeamMonitor::address_t const out_address ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_out_address)( this->getCApiPtr(), out_address );
        return;
    }

    SIXTRL_INLINE void BeamMonitor::setMinParticleId(
        BeamMonitor::index_t const min_particle_id ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_min_particle_id)(
            this->getCApiPtr(), min_particle_id );

        return;
    }

    SIXTRL_INLINE void BeamMonitor::setMaxParticleId(
        BeamMonitor::index_t const max_particle_id ) SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_set_max_particle_id)(
            this->getCApiPtr(), max_particle_id );

        return;
    }

    SIXTRL_INLINE void BeamMonitor::clear() SIXTRL_NOEXCEPT
    {
        ::NS(BeamMonitor_clear)( this->getCApiPtr() );
        return;
    }

    #if !defined( _GPUCODE )

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_new(
        Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        return static_cast< SIXTRL_BE_ARGPTR_DEC BeamMonitor* >(
            ::NS(BeamMonitor_new)( buffer.getCApiPtr() ) );
    }

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_new(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer )
    {
        return static_cast< SIXTRL_BE_ARGPTR_DEC BeamMonitor* >(
            ::NS(BeamMonitor_new)( buffer ) );
    }

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        BeamMonitor::turn_t    const num_stores,
        BeamMonitor::turn_t    const start,
        BeamMonitor::turn_t    const skip,
        BeamMonitor::address_t const out_address,
        BeamMonitor::index_t   const min_particle_id,
        BeamMonitor::index_t   const max_particle_id,
        bool const is_rolling, bool const is_turn_ordered )
    {
        return static_cast< SIXTRL_BE_ARGPTR_DEC BeamMonitor* >(
            ::NS(BeamMonitor_add)( buffer.getCApiPtr(), num_stores, start,
                skip, out_address, min_particle_id, max_particle_id,
                    is_rolling, is_turn_ordered ) );
    }

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        BeamMonitor::turn_t    const num_stores,
        BeamMonitor::turn_t    const start,
        BeamMonitor::turn_t    const skip,
        BeamMonitor::address_t const out_address,
        BeamMonitor::index_t   const min_particle_id,
        BeamMonitor::index_t   const max_particle_id,
        bool const is_rolling, bool const is_turn_ordered )
    {
        return static_cast< SIXTRL_BE_ARGPTR_DEC BeamMonitor* >(
            ::NS(BeamMonitor_add)( buffer, num_stores, start, skip,
                out_address, min_particle_id, max_particle_id,
                    is_rolling, is_turn_ordered ) );
    }

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        BeamMonitor const& SIXTRL_RESTRICT_REF orig )
    {
        return static_cast< SIXTRL_BE_ARGPTR_DEC BeamMonitor* >(
            ::NS(BeamMonitor_add_copy)(
                buffer.getCApiPtr(), orig.getCApiPtr() ) );
    }

    SIXTRL_BE_ARGPTR_DEC BeamMonitor* BeamMonitor_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT buffer,
        BeamMonitor const& SIXTRL_RESTRICT_REF orig )
    {
        return static_cast< SIXTRL_BE_ARGPTR_DEC BeamMonitor* >(
            ::NS(BeamMonitor_add_copy)( buffer, orig.getCApiPtr() ) );
    }


    SIXTRL_INLINE bool BeamMonitor_insert_end_of_turn_monitors(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        BeamMonitor::turn_t const turn_by_turn_start,
        BeamMonitor::turn_t const num_turn_by_turn_turns,
        BeamMonitor::turn_t const target_num_turns,
        BeamMonitor::turn_t const skip_turns,
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC Buffer::object_t* prev_node )
    {
        return ::NS(BeamMonitor_insert_end_of_turn_monitors)(
            beam_elements_buffer.getCApiPtr(), turn_by_turn_start,
            num_turn_by_turn_turns, target_num_turns, skip_turns, prev_node );
    }

    SIXTRL_INLINE bool BeamMonitor_insert_end_of_turn_monitors(
        Buffer& SIXTRL_RESTRICT_REF beam_elements_buffer,
        BeamMonitor::turn_t const turn_by_turn_start,
        BeamMonitor::turn_t const num_turn_by_turn_turns,
        BeamMonitor::turn_t const target_num_turns,
        BeamMonitor::turn_t const skip_turns,
        Buffer::size_type   const insert_at_index )
    {
        return ::NS(BeamMonitor_insert_end_of_turn_monitors_at_pos)(
            beam_elements_buffer.getCApiPtr(), turn_by_turn_start,
            num_turn_by_turn_turns, target_num_turns, skip_turns,
                insert_at_index );
    }

    SIXTRL_INLINE bool BeamMonitor_insert_end_of_turn_monitors(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(be_monitor_turn_t) const turn_by_turn_start,
        ::NS(be_monitor_turn_t) const num_turn_by_turn_turns,
        ::NS(be_monitor_turn_t) const target_num_turns,
        ::NS(be_monitor_turn_t) const skip_turns,
        SIXTRL_BUFFER_OBJ_ARGPTR_DEC ::NS(Object)* prev_node )
    {
        return ::NS(BeamMonitor_insert_end_of_turn_monitors)(
            belements_buffer, turn_by_turn_start, num_turn_by_turn_turns,
            target_num_turns, skip_turns, prev_node );
    }

    SIXTRL_INLINE bool BeamMonitor_insert_end_of_turn_monitors(
        SIXTRL_BUFFER_ARGPTR_DEC ::NS(Buffer)* SIXTRL_RESTRICT belements_buffer,
        ::NS(be_monitor_turn_t) const turn_by_turn_start,
        ::NS(be_monitor_turn_t) const num_turn_by_turn_turns,
        ::NS(be_monitor_turn_t) const target_num_turns,
        ::NS(be_monitor_turn_t) const skip_turns,
        ::NS(buffer_size_t)     const insert_at_index )
    {
        return ::NS(BeamMonitor_insert_end_of_turn_monitors_at_pos)(
            belements_buffer, turn_by_turn_start, num_turn_by_turn_turns,
            target_num_turns, skip_turns, insert_at_index );
    }

    #endif /* !defined( _GPUCODE ) */
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BE_MONITOR_BE_MONITOR_CXX_HEADER_HPP__ */

/* end: sixtracklib/common/be_monitor/be_monitor.hpp */
