#ifndef SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_CXX_HPP__
#define SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_CXX_HPP__

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
    #include "sixtracklib/common/control/definitions.h"
    #include "sixtracklib/common/buffer.hpp"
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/buffer/assign_address_item.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    struct AssignAddressItem : public ::NS(AssignAddressItem)
    {
        typedef SIXTRL_CXX_NAMESPACE::Buffer buffer_t;
        typedef buffer_t::c_api_t            c_buffer_t;
        typedef buffer_t::size_type          size_type;
        typedef buffer_t::type_id_t          type_id_t;
        typedef ::NS(AssignAddressItem)      c_api_t;

        SIXTRL_FN AssignAddressItem() SIXTRL_NOEXCEPT;

        SIXTRL_FN AssignAddressItem(
            type_id_t const dest_elem_type_id,
            size_type const dest_buffer_id,
            size_type const dest_elem_idx,
            size_type const dest_pointer_offset,
            type_id_t const src_elem_type_id,
            size_type const src_buffer_id,
            size_type const src_elem_idx,
            size_type const src_pointer_offset ) SIXTRL_NOEXCEPT;


        SIXTRL_FN AssignAddressItem(
            AssignAddressItem const& other ) = default;

        SIXTRL_FN AssignAddressItem& operator=(
            AssignAddressItem const& rhs ) = default;

        SIXTRL_FN AssignAddressItem( AssignAddressItem&& other ) = default;
        SIXTRL_FN AssignAddressItem& operator=(
            AssignAddressItem&& rhs ) = default;

        SIXTRL_FN ~AssignAddressItem() = default;

        /* ----------------------------------------------------------------- */

        #if !defined( _GPUCODE )

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem const* FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC Buffer const& SIXTRL_RESTRICT_REF buffer,
            size_type const buffer_index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem const* FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC const ::NS(Buffer) *const
                SIXTRL_RESTRICT buffer,
            size_type const buffer_index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem const* FromBufferObject(
            SIXTRL_BUFFER_ARGPTR_DEC SIXTRL_BUFFER_OBJ_ARGPTR_DEC const
                NS(Object) *const SIXTRL_RESTRICT item_info ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC Buffer& SIXTRL_RESTRICT_REF buffer,
            size_type const buffer_index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
            size_type const buffer_index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN
        SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* FromBufferObject(
            SIXTRL_BUFFER_OBJ_ARGPTR_DEC
                NS(Object)* SIXTRL_RESTRICT item_info ) SIXTRL_NOEXCEPT;

        #endif /* !defined( _GPUCODE ) */

        /* ----------------------------------------------------------------- */

        #if !defined( _GPUCODE )

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_objects  = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_slots    = nullptr,
            SIXTRL_BUFFER_ARGPTR_DEC size_type* SIXTRL_RESTRICT
                ptr_requ_dataptrs = nullptr ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
        CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
        AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            type_id_t const dest_elem_type_id,
            size_type const dest_buffer_id,
            size_type const dest_elem_idx,
            size_type const dest_pointer_offset,
            type_id_t const src_elem_type_id,
            size_type const src_buffer_id,
            size_type const src_elem_idx,
            size_type const src_pointer_offset );

        #endif /* !defined( _GPUCODE ) */

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumDataPtrs() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC c_api_t const*
            getCApiPtr() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC c_api_t*
            getCApiPtr() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;
        SIXTRL_FN bool valid() const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool is_on_buffer() const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool is_on_raw_memory() const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool dest_is_on_buffer() const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool src_is_on_buffer() const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool dest_is_on_raw_memory() const SIXTRL_NOEXCEPT;
        SIXTRL_FN bool src_is_on_raw_memory() const SIXTRL_NOEXCEPT;

        SIXTRL_FN type_id_t getDestElemTypeId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getDestBufferId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getDestElemIndex() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getDestElemPointerOffset() const SIXTRL_NOEXCEPT;

        SIXTRL_FN type_id_t getSrcElemTypeId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSrcBufferId() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSrcElemIndex() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSrcElemPointerOffset() const SIXTRL_NOEXCEPT;

        /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */

        SIXTRL_FN void setDestElemTypeId(
            type_id_t const dest_elem_type_id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setDestBufferId(
            size_type const dest_buffer_id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setDestElemIndex(
            size_type const dest_elem_idx ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setDestElemPointerOffset(
            size_type const dest_elem_pointer_offset ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setSrcElemTypeId(
            type_id_t const dest_elem_type_id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setSrcBufferId(
            size_type const dest_buffer_id ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setSrcElemIndex(
            size_type const dest_elem_idx ) SIXTRL_NOEXCEPT;

        SIXTRL_FN void setSrcElemPointerOffset(
            size_type const dest_elem_pointer_offset ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN bool operator<( AssignAddressItem const&
            SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool operator==( AssignAddressItem const&
            SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool operator!=( AssignAddressItem const&
            SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool operator>=( AssignAddressItem const&
            SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool operator>( AssignAddressItem const&
            SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN bool operator<=( AssignAddressItem const&
            SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT;
    };

    #if !defined( _GPUCODE )

    SIXTRL_STATIC SIXTRL_FN
    SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* AssignAddressItem_new(
        Buffer& SIXTRL_RESTRICT_REF buffer );

    SIXTRL_STATIC SIXTRL_FN
    SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* AssignAddressItem_new(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer );

    SIXTRL_STATIC SIXTRL_FN
    SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* AssignAddressItem_add(
        AssignAddressItem::buffer_t& SIXTRL_RESTRICT_REF buffer,
        AssignAddressItem::type_id_t const dest_elem_type_id,
        AssignAddressItem::size_type const dest_buffer_id,
        AssignAddressItem::size_type const dest_elem_idx,
        AssignAddressItem::size_type const dest_pointer_offset,
        AssignAddressItem::type_id_t const src_elem_type_id,
        AssignAddressItem::size_type const src_buffer_id,
        AssignAddressItem::size_type const src_elem_idx,
        AssignAddressItem::size_type const src_pointer_offset );

    SIXTRL_STATIC SIXTRL_FN
    SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* AssignAddressItem_add(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        AssignAddressItem::type_id_t const dest_elem_type_id,
        AssignAddressItem::size_type const dest_buffer_id,
        AssignAddressItem::size_type const dest_elem_idx,
        AssignAddressItem::size_type const dest_pointer_offset,
        AssignAddressItem::type_id_t const src_elem_type_id,
        AssignAddressItem::size_type const src_buffer_id,
        AssignAddressItem::size_type const src_elem_idx,
        AssignAddressItem::size_type const src_pointer_offset );

    SIXTRL_STATIC SIXTRL_FN
    SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* AssignAddressItem_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        AssignAddressItem const& SIXTRL_RESTRICT_REF orig );

    SIXTRL_STATIC SIXTRL_FN
    SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem* AssignAddressItem_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        AssignAddressItem const& SIXTRL_RESTRICT_REF orig );

    #endif /* !defined( _GPUCODE ) */

    template<> struct ObjectTypeTraits< ::NS(AssignAddressItem) >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM);
        }
    };

    template<> struct ObjectTypeTraits< AssignAddressItem >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM);
        }
    };
}

/* ************************************************************************* */
/* ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE AssignAddressItem::AssignAddressItem() SIXTRL_NOEXCEPT :
        ::NS(AssignAddressItem)()
    {
        ::NS(AssignAddressItem_preset)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::AssignAddressItem(
        AssignAddressItem::type_id_t const dest_elem_type_id,
        AssignAddressItem::size_type const dest_buffer_id,
        AssignAddressItem::size_type const dest_elem_idx,
        AssignAddressItem::size_type const dest_pointer_offset,
        AssignAddressItem::type_id_t const src_elem_type_id,
        AssignAddressItem::size_type const src_buffer_id,
        AssignAddressItem::size_type const src_elem_idx,
        AssignAddressItem::size_type const src_pointer_offset
            ) SIXTRL_NOEXCEPT : ::NS(AssignAddressItem)()
    {
        ::NS(AssignAddressItem_set_dest_elem_type_id)(
            this->getCApiPtr(), dest_elem_type_id );

        ::NS(AssignAddressItem_set_dest_buffer_id)(
            this->getCApiPtr(), dest_buffer_id );

        ::NS(AssignAddressItem_set_dest_elem_index)(
            this->getCApiPtr(), dest_elem_idx );

        ::NS(AssignAddressItem_set_dest_pointer_offset)(
            this->getCApiPtr(), dest_pointer_offset );

        ::NS(AssignAddressItem_set_src_elem_type_id)(
            this->getCApiPtr(), src_elem_type_id );

        ::NS(AssignAddressItem_set_src_buffer_id)(
            this->getCApiPtr(), src_buffer_id );

        ::NS(AssignAddressItem_set_src_elem_index)(
            this->getCApiPtr(), src_elem_idx );

        ::NS(AssignAddressItem_set_src_pointer_offset)(
            this->getCApiPtr(), src_pointer_offset );
    }

    #if !defined( _GPUCODE )

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem const*
    AssignAddressItem::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC Buffer const&
        SIXTRL_RESTRICT_REF buffer,
        AssignAddressItem::size_type const buffer_index ) SIXTRL_NOEXCEPT
    {
        return AssignAddressItem::FromBufferObject( buffer[ buffer_index ] );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem const*
    AssignAddressItem::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC const
        ::NS(Buffer) *const SIXTRL_RESTRICT buffer,
        AssignAddressItem::size_type const buffer_index ) SIXTRL_NOEXCEPT
    {
        return AssignAddressItem::FromBufferObject(
            ::NS(Buffer_get_const_object)( buffer, buffer_index ) );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem const*
    AssignAddressItem::FromBufferObject(
        SIXTRL_BUFFER_ARGPTR_DEC SIXTRL_BUFFER_OBJ_ARGPTR_DEC
            const ::NS(Object) *const SIXTRL_RESTRICT item_info ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem const*;

        return ( ( item_info != nullptr ) &&
                 ( ::NS(Object_get_type_id)( item_info ) ==
                    ::NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM) ) &&
                 ( ::NS(Object_get_size)( item_info ) >=
                    sizeof( AssignAddressItem ) ) )
            ? reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( item_info ) ) )
            : nullptr;
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem::FromBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC Buffer& SIXTRL_RESTRICT_REF buffer,
        AssignAddressItem::size_type const buffer_index ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        return const_cast< ptr_t >( AssignAddressItem::FromBufferObject(
            ::NS(Buffer_get_const_object)(
                buffer.getCApiPtr(), buffer_index ) ) );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem::FromBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
        AssignAddressItem::size_type const buffer_index ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        return const_cast< ptr_t >( AssignAddressItem::FromBufferObject(
            ::NS(Buffer_get_const_object)( buffer, buffer_index ) ) );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem::FromBufferObject( SIXTRL_BUFFER_OBJ_ARGPTR_DEC
        NS(Object)* SIXTRL_RESTRICT item_info ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        using ptr_c_obj_t = SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*;

        ptr_c_obj_t c_item_info = item_info;

        return const_cast< ptr_t >( AssignAddressItem::FromBufferObject(
            c_item_info ) );
    }

    #endif /* !defined( _GPUCODE ) */

    /* ----------------------------------------------------------------- */

    #if !defined( _GPUCODE )

    SIXTRL_INLINE bool AssignAddressItem::CanAddToBuffer(
        AssignAddressItem::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_ARGPTR_DEC AssignAddressItem::size_type*
            SIXTRL_RESTRICT ptr_requ_objects,
        SIXTRL_BUFFER_ARGPTR_DEC AssignAddressItem::size_type*
            SIXTRL_RESTRICT ptr_requ_slots,
        SIXTRL_BUFFER_ARGPTR_DEC AssignAddressItem::size_type*
            SIXTRL_RESTRICT ptr_requ_dataptrs ) SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_can_be_added)( buffer.getCApiPtr(),
                ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem::CreateNewOnBuffer( AssignAddressItem::buffer_t&
        SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        return static_cast< ptr_t >( ::NS(AssignAddressItem_new)(
            buffer.getCApiPtr() ) );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem::AddToBuffer(
        AssignAddressItem::buffer_t& SIXTRL_RESTRICT_REF buffer,
        AssignAddressItem::type_id_t const dest_elem_type_id,
        AssignAddressItem::size_type const dest_buffer_id,
        AssignAddressItem::size_type const dest_elem_idx,
        AssignAddressItem::size_type const dest_pointer_offset,
        AssignAddressItem::type_id_t const src_elem_type_id,
        AssignAddressItem::size_type const src_buffer_id,
        AssignAddressItem::size_type const src_elem_idx,
        AssignAddressItem::size_type const src_pointer_offset )
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        return static_cast< ptr_t >( ::NS(AssignAddressItem_add)(
            buffer.getCApiPtr(), dest_elem_type_id, dest_buffer_id,
                    dest_elem_idx, dest_pointer_offset,
            src_elem_type_id, src_buffer_id,
                    src_elem_idx, src_pointer_offset ) );
    }

    #endif /* !defined( _GPUCODE ) */

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE AssignAddressItem::type_id_t
    AssignAddressItem::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_type_id)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::size_type
    AssignAddressItem::getNumDataPtrs() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_num_dataptrs)( this->getCApiPtr() );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem::c_api_t const*
    AssignAddressItem::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast<
            SIXTRL_BUFFER_DATAPTR_DEC c_api_t const* >( this );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem::c_api_t*
    AssignAddressItem::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< SIXTRL_BUFFER_DATAPTR_DEC c_api_t* >( this );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE void AssignAddressItem::preset() SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_preset)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool AssignAddressItem::valid() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_valid)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool AssignAddressItem::is_on_buffer() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_is_on_buffer)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool
    AssignAddressItem::is_on_raw_memory() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_is_on_raw_memory)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool
    AssignAddressItem::dest_is_on_buffer() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_dest_is_on_buffer)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool
    AssignAddressItem::src_is_on_buffer() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_src_is_on_buffer)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool
    AssignAddressItem::dest_is_on_raw_memory() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_dest_is_on_raw_memory)(
            this->getCApiPtr() );
    }

    SIXTRL_INLINE bool
    AssignAddressItem::src_is_on_raw_memory() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_src_is_on_raw_memory)(
            this->getCApiPtr() );
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE AssignAddressItem::type_id_t
    AssignAddressItem::getDestElemTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_dest_elem_type_id)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::size_type
    AssignAddressItem::getDestBufferId() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_dest_buffer_id)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::size_type
    AssignAddressItem::getDestElemIndex() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_dest_elem_index)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::size_type
    AssignAddressItem::getDestElemPointerOffset() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_dest_pointer_offset)(
            this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::type_id_t
    AssignAddressItem::getSrcElemTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_src_elem_type_id)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::size_type
    AssignAddressItem::getSrcBufferId() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_src_buffer_id)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::size_type
    AssignAddressItem::getSrcElemIndex() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_src_elem_index)( this->getCApiPtr() );
    }

    SIXTRL_INLINE AssignAddressItem::size_type
    AssignAddressItem::getSrcElemPointerOffset() const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_src_pointer_offset)( this->getCApiPtr() );
    }

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- */

    SIXTRL_INLINE void AssignAddressItem::setDestElemTypeId(
        AssignAddressItem::type_id_t const dest_elem_type_id ) SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_set_dest_elem_type_id)(
            this->getCApiPtr(), dest_elem_type_id );
    }

    SIXTRL_INLINE void AssignAddressItem::setDestBufferId(
        AssignAddressItem::size_type const dest_buffer_id ) SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_set_dest_buffer_id)(
            this->getCApiPtr(), dest_buffer_id );
    }

    SIXTRL_INLINE void AssignAddressItem::setDestElemIndex(
        AssignAddressItem::size_type const dest_elem_idx ) SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_set_dest_elem_index)(
            this->getCApiPtr(), dest_elem_idx );
    }

    SIXTRL_INLINE void AssignAddressItem::setDestElemPointerOffset(
        AssignAddressItem::size_type const
            dest_elem_pointer_offset ) SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_set_dest_pointer_offset)(
            this->getCApiPtr(), dest_elem_pointer_offset );
    }

    SIXTRL_INLINE void AssignAddressItem::setSrcElemTypeId(
        AssignAddressItem::type_id_t const src_elem_type_id ) SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_set_src_elem_type_id)(
            this->getCApiPtr(), src_elem_type_id );
    }

    SIXTRL_INLINE void AssignAddressItem::setSrcBufferId(
        AssignAddressItem::size_type const src_buffer_id ) SIXTRL_NOEXCEPT
    {
         ::NS(AssignAddressItem_set_src_buffer_id)(
            this->getCApiPtr(), src_buffer_id );
    }

    SIXTRL_INLINE void AssignAddressItem::setSrcElemIndex(
        AssignAddressItem::size_type const src_elem_idx ) SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_set_src_elem_index)(
            this->getCApiPtr(), src_elem_idx );
    }

    SIXTRL_INLINE void AssignAddressItem::setSrcElemPointerOffset(
        AssignAddressItem::size_type const
            dest_elem_pointer_offset ) SIXTRL_NOEXCEPT
    {
        ::NS(AssignAddressItem_set_src_pointer_offset)(
            this->getCApiPtr(), dest_elem_pointer_offset );
    }

     /* ----------------------------------------------------------------- */

    SIXTRL_INLINE bool AssignAddressItem::operator<( AssignAddressItem const&
        SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_compare_less)(
            this->getCApiPtr(), rhs.getCApiPtr() );
    }

    SIXTRL_INLINE bool AssignAddressItem::operator==( AssignAddressItem const&
        SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_are_equal)(
            this->getCApiPtr(), rhs.getCApiPtr() );
    }

    SIXTRL_INLINE bool AssignAddressItem::operator!=( AssignAddressItem const&
        SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT
    {
        return !::NS(AssignAddressItem_compare_less)(
            this->getCApiPtr(), rhs.getCApiPtr() );
    }

    SIXTRL_INLINE bool AssignAddressItem::operator>=( AssignAddressItem const&
        SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT
    {
        return ::NS(AssignAddressItem_compare_less)(
            this->getCApiPtr(), rhs.getCApiPtr() );
    }

    SIXTRL_INLINE bool AssignAddressItem::operator>( AssignAddressItem const&
        SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT
    {
        return ( ( !::NS(AssignAddressItem_are_equal)(
                     this->getCApiPtr(), rhs.getCApiPtr() ) ) &&
                 ( !::NS(AssignAddressItem_compare_less)(
                     this->getCApiPtr(), rhs.getCApiPtr() ) ) );
    }

    SIXTRL_INLINE bool AssignAddressItem::operator<=( AssignAddressItem const&
        SIXTRL_RESTRICT_REF rhs ) const SIXTRL_NOEXCEPT
    {
        return ( ( ::NS(AssignAddressItem_are_equal)(
                     this->getCApiPtr(), rhs.getCApiPtr() ) ) ||
                 ( ::NS(AssignAddressItem_compare_less)(
                     this->getCApiPtr(), rhs.getCApiPtr() ) ) );
    }

    /* --------------------------------------------------------------------- */

    #if !defined( _GPUCODE )

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem_new( Buffer& SIXTRL_RESTRICT_REF buffer )
    {
        return AssignAddressItem::CreateNewOnBuffer( buffer );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem_new(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer )
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        return static_cast< ptr_t >(
            ::NS(AssignAddressItem_new)( ptr_buffer ) );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem_add(
        AssignAddressItem::buffer_t& SIXTRL_RESTRICT_REF buffer,
        AssignAddressItem::type_id_t const dest_elem_type_id,
        AssignAddressItem::size_type const dest_buffer_id,
        AssignAddressItem::size_type const dest_elem_idx,
        AssignAddressItem::size_type const dest_pointer_offset,
        AssignAddressItem::type_id_t const src_elem_type_id,
        AssignAddressItem::size_type const src_buffer_id,
        AssignAddressItem::size_type const src_elem_idx,
        AssignAddressItem::size_type const src_pointer_offset )
    {
        return AssignAddressItem::AddToBuffer( buffer, dest_elem_type_id,
            dest_buffer_id, dest_elem_idx, dest_pointer_offset,
            src_elem_type_id, src_buffer_id, src_elem_idx, src_pointer_offset );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem_add(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        AssignAddressItem::type_id_t const dest_elem_type_id,
        AssignAddressItem::size_type const dest_buffer_id,
        AssignAddressItem::size_type const dest_elem_idx,
        AssignAddressItem::size_type const dest_pointer_offset,
        AssignAddressItem::type_id_t const src_elem_type_id,
        AssignAddressItem::size_type const src_buffer_id,
        AssignAddressItem::size_type const src_elem_idx,
        AssignAddressItem::size_type const src_pointer_offset )
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        return static_cast< ptr_t >( ::NS(AssignAddressItem_add)(
            ptr_buffer, dest_elem_type_id, dest_buffer_id,
                dest_elem_idx, dest_pointer_offset,
            src_elem_type_id, src_buffer_id, src_elem_idx,
                src_pointer_offset ) );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem_add_copy(
        Buffer& SIXTRL_RESTRICT_REF buffer,
        AssignAddressItem const& SIXTRL_RESTRICT_REF orig )
    {
        return SIXTRL_CXX_NAMESPACE::AssignAddressItem_add(
            buffer, orig.getDestElemTypeId(), orig.getDestBufferId(),
            orig.getDestElemIndex(), orig.getDestElemPointerOffset(),
            orig.getSrcElemTypeId(), orig.getSrcBufferId(),
            orig.getSrcElemIndex(),  orig.getSrcElemPointerOffset() );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*
    AssignAddressItem_add_copy(
        SIXTRL_BUFFER_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT ptr_buffer,
        AssignAddressItem const& SIXTRL_RESTRICT_REF orig )
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC AssignAddressItem*;
        return static_cast< ptr_t >( ::NS(AssignAddressItem_add_copy)(
            ptr_buffer, orig.getCApiPtr() ) );
    }

    #endif /* !defined( _GPUCODE ) */
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_ASSIGN_ADDRESS_ITEM_CXX_HPP__ */
