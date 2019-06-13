#ifndef SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_HPP__
#define SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdlib>
    #if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
        #include <string>
    #endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */
    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/buffer/buffer_array_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class BufferArrayObj : public ::NS(BufferArrayObj)
    {
        public:

        using c_api_t       = ::NS(BufferArrayObj);
        using buffer_t      = SIXTRL_CXX_NAMESPACE::Buffer;
        using size_type     = buffer_t::size_type;
        using address_t     = buffer_t::address_t;
        using c_buffer_t    = buffer_t::c_api_t;
        using object_t      = buffer_t::object_t;
        using type_id_t     = buffer_t::type_id_t;
        using ptr_stored_t  = SIXTRL_BUFFER_DATAPTR_DEC BufferArrayObj*;

        using ptr_const_stored_t =
            SIXTRL_BUFFER_DATAPTR_DEC BufferArrayObj const*;

        using ptr_size_arg_t =  SIXTRL_BUFFER_ARGPTR_DEC size_type*;
        using ptr_c_api_t    =  SIXTRL_BUFFER_DATAPTR_DEC BufferArrayObj*;
        using ptr_object_t   =  SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t*;

        using ptr_const_object_t =
            SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t const*;

        SIXTRL_FN explicit BufferArrayObj(
            address_t const data_begin_addr = address_t{ 0 },
            address_t const offset_list_begin_addr = address_t{ 0 },
            size_type const num_elements = size_type{ 0 },
            size_type const max_num_elements = size_type{ 0 },
            size_type const capacity = size_type{ 0 },
            size_type const slot_size = size_type{ 0 },
            type_id_t const base_type_id = ::NS(OBJECT_TYPE_UNDEFINED)
        ) SIXTRL_NOEXCEPT;

        SIXTRL_FN BufferArrayObj( BufferArrayObj const& other ) = default;
        SIXTRL_FN BufferArrayObj( BufferArrayObj&& other ) = default;

        SIXTRL_FN BufferArrayObj&
        operator=( BufferArrayObj const& rhs ) = default;

        SIXTRL_FN BufferArrayObj&
        operator=( BufferArrayObj&& rhs ) = default;

        SIXTRL_FN ~BufferArrayObj() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_FN ptr_const_stored_t FromBuffer(
            Buffer const& SIXTRL_RESTRICT_REF buffer,
            size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN ptr_const_stored_t FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC const c_buffer_t *const SIXTRL_RESTRICT
                buffer, size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t FromBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t FromBuffer(
            SIXTRL_BUFFER_ARGPTR_DEC c_buffer_t* SIXTRL_RESTRICT_REF buffer,
            size_type const index ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN ptr_const_stored_t FromBufferObject(
            ptr_const_object_t SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t FromBufferObject(
            ptr_object_t SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN bool CanAddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const max_nelements, size_type const capacity,
            ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_objects  = nullptr,
            ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_slots    = nullptr,
            ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_dataptrs = nullptr
        ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const max_nelements, size_type const capacity );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            address_t const data_begin_address,
            address_t const offset_list_begin_addr,
            size_type const num_elements,
            size_type const max_nelements,
            size_type const capacity,
            type_id_t const base_type_id,
            size_type const slot_size = buffer_t::DEFAULT_SLOT_SIZE );

        template< typename Iter >
        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_BUFFER_DATAPTR_DEC const void *const
                SIXTRL_RESTRICT data_buffer,
            Iter offset_begin, Iter offset_end,
            size_type const max_nelements, size_type const capacity,
            type_id_t const base_type_id,
            size_type const slot_size = buffer_t::DEFAULT_SLOT_SIZE );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t CreateNewOnBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const max_nelements, size_type const capacity );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            address_t const data_begin_address,
            address_t const offset_list_begin_addr,
            size_type const num_elements,
            size_type const max_nelements,
            size_type const capacity,
            type_id_t const base_type_id,
            size_type const slot_size = buffer_t::DEFAULT_SLOT_SIZE );

        template< typename Iter >
        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_BUFFER_DATAPTR_DEC const void *const
                SIXTRL_RESTRICT data_buffer,
            Iter offsets_begin, Iter offsets_end,
            size_type const max_nelements, size_type const capacity,
            type_id_t const base_type_id,
            size_type const slot_size = buffer_t::DEFAULT_SLOT_SIZE );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumDataPtrs(
            size_type const max_nelements = size_type{ 0 },
            size_type const capacity = size_type{ 0 },
            size_type const slot_size = buffer_t::DEFAULT_SLOT_SIZE
        ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumSlots(
                size_type const max_nelements = size_type{ 0 },
                size_type const capacity = size_type{ 0 },
                size_type const slot_size = buffer_t::DEFAULT_SLOT_SIZE
        ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN address_t getDataBeginAddress() const SIXTRL_NOEXCEPT;
        SIXTRL_FN address_t getOffsetsListBeginAddress() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getNumElements() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getMaxNumElements() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getCapacity() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getSlotSize() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_Type getBaseTypeId() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void const*
        constDataBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void const*
        constDataEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void* dataBegin() SIXTRL_NOEXCEPT;
        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC void* dataEnd() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC ŝize_type const*
        constElementOffsetListBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_FN SIXTRL_BUFFER_DATAPTR_DEC ŝize_type const*
        constElementOffsetListEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_FN ŝize_type elementOffset(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN address_t getElementBeginAddress(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getElementLength(
            size_type const index ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN address_t getElementEndAddress(
            size_type const index ) const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN void clear() SIXTRL_NOEXCEPT;

        template< typename T >
        SIXTRL_FN bool append(
            T& SIXTRL_RESTRICT_REF obj_handle ) SIXTRL_NOEXCEPT;

        template< typename T >
        SIXTRL_FN bool push_back(
            T& SIXTRL_RESTRICT_REF obj_handle ) SIXTRL_NOEXCEPT;

        SIXTRL_FN bool removeLastElement() SIXTRL_NOEXCEPT;

        SIXTRL_FN bool pop_back() SIXTRL_NOEXCEPT;

        SIXTRL_FN bool removeLastNumElements(
            size_type const num_elements_to_remove ) SIXTRL_NOEXCEPT;
    };

    /* --------------------------------------------------------------------- */

    template<> struct ObjectTypeTraits< ::NS(BufferArrayObj) >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_ARRAY);
        }
    };

    template<> struct ObjectTypeTraits< SIXTRL_CXX_NAMESPACE::BufferArrayObj >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_ARRAY;
        }
    };

    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_new(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        size_type const max_nelements, size_type const capacity )
    {

    }

    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_new(
        SIXTRL_BUFFER_ARGPTR_DEC BufferArrayObj::c_buffer_t&
            SIXTRL_RESTRICT_REF buffer,
        size_type const max_nelements, size_type const capacity ); );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::address_t const data_begin_addr,
        BufferArrayObj::address_t const offset_list_begin_addr,
        BufferArrayObj::size_type const num_elements,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size =
            BufferArrayObj::buffer_t::DEFAULT_SLOT_SIZE );

    template< typename Iter >
    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_DATAPTR_DEC const void *const
                SIXTRL_RESTRICT data_buffer,
        Iter offset_begin, Iter offset_end,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size =
            BufferArrayObj::buffer_t::DEFAULT_SLOT_SIZE );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::address_t const data_begin_addr,
        BufferArrayObj::address_t const offset_list_begin_addr,
        BufferArrayObj::size_type const num_elements,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size =
            BufferArrayObj::buffer_t::DEFAULT_SLOT_SIZE );

    template< typename Iter >
    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_DATAPTR_DEC const void *const
                SIXTRL_RESTRICT data_buffer,
        Iter offset_begin, Iter offset_end,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size =
            BufferArrayObj::buffer_t::DEFAULT_SLOT_SIZE );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add_copy(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj const& SIXTRL_RESTRICT_REF orig );

    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add_copy(
        BufferArrayObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj const& SIXTRL_RESTRICT_REF orig );
}

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <algorithm>
    #include <iterator>
#endif /*

/* ************************************************************************* */
/* ****** ******* */
/* ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE  BufferArrayObj::BufferArrayObj(
        BufferArrayObj::address_t const data_begin_addr,
        BufferArrayObj::address_t const offset_list_begin_addr,
        BufferArrayObj::size_type const num_elements ,
        BufferArrayObj::size_type const max_num_elements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::size_type const slot_size,
        BufferArrayObj::type_id_t const base_type_id ) SIXTRL_NOEXCEPT
    {

    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE BufferArrayObj::ptr_const_stored_t
    BufferArrayObj::FromBuffer(
        BufferArrayObj::buffer_t const& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        return BufferArrayObj::FromBufferObject( buffer[ index ] );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_const_stored_t
    BufferArrayObj::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC const
        BufferArrayObj::c_buffer_t *const SIXTRL_RESTRICT buffer,
        BufferArrayObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        return BufferArrayObj::FromBufferObject(
            ::NS(Buffer_get_const_object)( buffer, index ) );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj::FromBuffer(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        return BufferArrayObj::FromBufferObject( buffer[ index ] );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj::FromBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC BufferArrayObj::c_buffer_t*
            SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        return BufferArrayObj::FromBufferObject(
            ::NS(Buffer_get_object)( buffer, index ) );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_const_stored_t
    BufferArrayObj::FromBufferObject( BufferArrayObj::ptr_const_object_t
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
    {
        using ptr_t = BufferArrayObj::ptr_const_stored_t;

        ptr_t ptr_array_obj = nullptr;

        if( ( ::NS(Object_get_type_id)( obj ) == ::NS(OBJECT_TYPE_ARRAY) ) &&
            ( ::NS(Object_get_size)( obj ) > sizeof( ::NS(BufferArrayObj) ) ) )
        {
            ptr_array_obj = reinterpret_cast< ptr_t >( static_cast< uintptr_t
                >( ::NS(Object_get_begin_addr)( obj ) ) );
        }

        return ptr_array_obj;
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t
    BufferArrayObj::FromBufferObject(
        BufferArrayObj::ptr_object_t SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
    {
        BufferArrayObj::ptr_const_object_t cobj = obj;
        return const_cast< BufferArrayObj::ptr_stored_t >(
            BufferArrayObj::FromBufferObject( cobj ) );
    }

    SIXTRL_INLINE bool BufferArrayObj::CanAddToBuffer(
            BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
            BufferArrayObj::size_type const max_nelements,
            BufferArrayObj::size_type const capacity,
            BufferArrayObj::ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_objects,
            BufferArrayObj::ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_slots,
            BufferArrayObj::ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_dataptrs
        ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_can_be_added)( *buffer.getCApiPtr(),
            max_nelements, capacity, ptr_requ_objects, ptr_requ_slots,
                ptr_requ_dataptrs );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t
    BufferArrayObj::CreateNewOnBuffer(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id )
    {
        return BufferArrayObj::CreateNewOnBuffer( *buffer.getCApiPtr(),
            max_nelements, capacity, base_type_id );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj::AddToBuffer(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::address_t const data_begin_address,
        BufferArrayObj::address_t const offset_list_begin_addr,
        BufferArrayObj::size_type const num_elements,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size )
    {
        return BufferArrayObj::AddToBuffer( *buffer.getCApiPtr(),
            data_begin_address, offset_list_begin_addr, num_elements,
                max_nelements, capacity, base_type_id, slot_size );
    }

    template< typename Iter >
    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj::AddToBuffer(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_DATAPTR_DEC const void *const
            SIXTRL_RESTRICT data_buffer,
        Iter offset_begin, Iter offset_end,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size )
    {
        return BufferArrayObj::AddToBuffer( *buffer.getCApiPtr(), data_buffer,
            offset_begin, offset_end, max_nelements, capacity, base_type_id,
                slot_size );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t
    BufferArrayObj::CreateNewOnBuffer(
        BufferArrayObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id )
    {
        return reinterpret_cast< BufferArrayObj::ptr_stored_t >(
            ::NS(BufferArrayObj_new)( &buffer, max_nelements,
                capacity, base_type_id ) );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj::AddToBuffer(
        BufferArrayObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::address_t const data_begin_address,
        BufferArrayObj::address_t const offset_list_begin_addr,
        BufferArrayObj::size_type const num_elements,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size )
    {
        return reinterpret_cast< BufferArrayObj::ptr_stored_t >(
            ::NS(BufferArrayObj_add)( &buffer, data_begin_addr,
                offset_list_begin_addr, num_elements, max_nelements, capacity,
                    base_type_id, slot_size ) );
    }

    template< typename Iter > SIXTRL_INLINE BufferArrayObj::ptr_stored_t
    BufferArrayObj::AddToBuffer(
        BufferArrayObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_DATAPTR_DEC const void *const SIXTRL_RESTRICT data_buffer,
        Iter offsets_begin, Iter offsets_end,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size )
    {
        using diff_t = ::ptrdiff_t;
        using size_t = BufferArrayObj::size_type;

        BufferArrayObj::ptr_stored_t ptr_array_obj = nullptr;

        diff_t const temp_len = std::distance( offset_begin, offset_end );

        if( temp_len >= diff_t{ 0 } )
        {
            ptr_array_obj = BufferArrayObj::CreateNewOnBuffer(
                buffer, max_nelements, capacity, base_type_id );

            if( ptr_array_obj != nullptr )
            {
                size_t const num_offsets = static_cast< size_t >( temp_len );

                if( ( slot_size != ::NS(BufferArrayObj_get_slot_size)(
                        ptr_array_obj ) ) &&
                    ( slot_size > size_t{ 0 } ) )
                {
                    ::NS(BufferArrayObj_set_slot_size)(
                        ptr_array_obj, slot_size );
                }

                SIXTRL_ASSERT( ( num_offsets == size_t{ 0 } ) ||
                    ( std::is_sorted( offset_begin, offset_end ) ) );

                if( num_offsets > size_t{ 0 } )
                {
                    size_t const num_elements = num_offsets - size_t{ 1 };
                    Iter it = offset_begin;

                    if( data_buffer != nullptr )
                    {
                        std::advance( it, num_elements );
                        buf_size_t const last_offset = *it;
                        SIXTRL_ASSERT( last_offset <= this->getCapacity() );

                        unsigned char const* in_begin = reinterpret_cast<
                            unsigned char const* >( data_buffer );

                        unsigned char const* in_end = in_begin;

                        unsigned char* out_begin = reinterpret_cast<
                            unsigned char* >( this->dataBegin() );

                        std::advance( in_end, last_offset );
                        std::copy( in_begin, in_end, out_begin );

                        it = offset_begin;
                    }

                    Iter prev = it++;

                    for( ; it != offset_end ; prev = it++ )
                    {
                        SIXTRL_ASSERT( *it >= *prev );
                        size_t const length = *it - *prev;

                        bool const ret = ::SN(BufferArrayObj_append_element)(
                            this->getCApiPtr(), nullptr, length );

                        SIXTRL_ASSERT( ret );
                        ( void )ret;
                    }

                    SIXTRL_ASSERT( this->getNumElements() == num_elements );
                }
            }
        }

        return ptr_array_obj;
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE BufferArrayObj::type_id_t
    BufferArrayObj::getTypeId() const SIXTRL_NOEXCEPT
    {
        return SIXTRL_CXX_NAMESPACE::OBJECT_TYPE_ARRAY;
    }

    SIXTRL_INLINE BufferArrayObj::size_type BufferArrayObj::getNumDataPtrs(
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::size_type const slot_size ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_required_num_dataptrs_on_managed_buffer)(
            max_nelements, capacity, slot_size );
    }

    SIXTRL_INLINE BufferArrayObj::size_type BufferArrayObj::getNumSlots(
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::size_type const slot_size ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_required_num_slots_on_managed_buffer)(
            max_nelements, capacity, slot_size );
    }

    SIXTRL_INLINE BufferArrayObj::c_api_t const*
    BufferArrayObj::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return static_cast< BufferArrayObj::c_api_t const* >( this );
    }

    SIXTRL_INLINE BufferArrayObj::c_api_t*
    BufferArrayObj::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< BufferArrayObj::c_api_t* >(
            static_cast< BufferArrayObj const& >( *this ).getCApiPtr() );
    }

    SIXTRL_INLINE void BufferArrayObj::preset() SIXTRL_NOEXCEPT
    {
        ::NS(BufferArrayObj_preset)( this->getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE BufferArrayObj::address_t
    BufferArrayObj::getDataBeginAddress() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_begin_addr)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferArrayObj::address_t
    BufferArrayObj::getOffsetsListBeginAddress() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_element_offset_list_begin_addr)(
            this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferArrayObj::size_type
    BufferArrayObj::getNumElements() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_num_elements)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferArrayObj::size_type
    BufferArrayObj::getMaxNumElements() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_max_num_elements)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferArrayObj::size_type
    BufferArrayObj::getLength() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferArrayObj::size_type
    BufferArrayObj::getCapacity() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_capacity)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferArrayObj::size_type
    BufferArrayObj::getSlotSize() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_slot_size)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferArrayObj::type_id_t
    BufferArrayObj::getBaseTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_type_id)( this->getCApiPtr() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void const*
    BufferArrayObj::constDataBegin() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_const_data_begin)( this->getCApiPtr() );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void const*
    BufferArrayObj::constDataEnd() const SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC void const*;
        ptr_t end_ptr = this->constDataBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->getLength() );
        }

        return end_ptr;
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void*
    BufferArrayObj::dataBegin() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BUFFER_DATAPTR_DEC void* >(
            this->constDataBegin() );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC void*
    BufferArrayObj::dataEnd() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BUFFER_DATAPTR_DEC void* >(
            this->constDataEnd() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC BufferArrayObj::ŝize_type const*
    BufferArrayObj::constElementOffsetListBegin() const SIXTRL_NOEXCEPT
    {
        using _this_t = BufferArrayObj;
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC _this_t::ŝize_type const*;

        return reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
            ::NS(BufferArrayObj_get_element_offset_list_begin_addr)(
                this->getCApiPtr() ) ) );
    }

    SIXTRL_INLINE SIXTRL_BUFFER_DATAPTR_DEC BufferArrayObj::ŝize_type const*
    BufferArrayObj::constElementOffsetListEnd() const SIXTRL_NOEXCEPT
    {
        using _this_t = BufferArrayObj;
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC _this_t::ŝize_type const*;

        ptr_t end_ptr = this->constElementOffsetListBegin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->getNumElements() );
        }

        return end_ptr;
    }

    SIXTRL_INLINE BufferArrayObj::ŝize_type BufferArrayObj::elementOffset(
        BufferArrayObj::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_element_offset)(
            this->getCApiPtr(), index );
    }

    SIXTRL_INLINE BufferArrayObj::address_t
    BufferArrayObj::getElementBeginAddress(
        BufferArrayObj::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_element_begin_addr)(
            this->getCApiPtr(), index );
    }

    SIXTRL_INLINE BufferArrayObj::size_type BufferArrayObj::getElementLength(
        BufferArrayObj::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_element_length)(
            this->getCApiPtr(), index );
    }

    SIXTRL_INLINE BufferArrayObj::address_t getElementEndAddress(
        BufferArrayObj::size_type const index ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_get_element_end_addr)(
            this->getCApiPtr(), index );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE void BufferArrayObj::clear() SIXTRL_NOEXCEPT
    {
        ::NS(BufferArrayObj_clear)( this->getCApiPtr() );
    }

    template< typename T > SIXTRL_INLINE bool BufferArrayObj::append(
        T& SIXTRL_RESTRICT_REF obj_handle ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_append_element)(
            this->getCApiPtr(), &obj_handle, sizeof( obj_handle ) );
    }

    template< typename T > SIXTRL_INLINE bool BufferArrayObj::push_back(
        T& SIXTRL_RESTRICT_REF obj_handle ) SIXTRL_NOEXCEPT
    {
        return this->append( obj_handle );
    }

    SIXTRL_INLINE bool BufferArrayObj::removeLastElement() SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_remove_last_element)( this->getCApiPtr() );
    }

    SIXTRL_INLINE bool BufferArrayObj::pop_back() SIXTRL_NOEXCEPT
    {
        return this->removeLastElement();
    }

    SIXTRL_INLINE bool BufferArrayObj::removeLastNumElements(
        size_type const num_elements_to_remove ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferArrayObj_remove_last_num_elements)(
            this->getCApiPtr(), num_elements_to_remove );
    }
}

/* ************************************************************************* */
/* ****** ******* */
/* ************************************************************************* */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj_new(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id )
    {
        return BufferArrayObj::CreateNewOnBuffer(
            *buffer.getCApiPtr(), max_nelements, capacity, base_type_id );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj_new(
        SIXTRL_BUFFER_ARGPTR_DEC BufferArrayObj::c_buffer_t&
            SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id )
    {
        return BufferArrayObj::CreateNewOnBuffer(
            &buffer, max_nelements, capacity, base_type_id );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::address_t const data_begin_addr,
        BufferArrayObj::address_t const offset_list_begin_addr,
        BufferArrayObj::size_type const num_elements,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size )
    {
        return BufferArrayObj::AddToBuffer( *buffer.getCApiPtr(),
            data_begin_addr, offset_list_begin_addr, num_elements,
                max_nelements, capacity, base_type_id, slot_size);
    }

    template< typename Iter > BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_DATAPTR_DEC const void *const
            SIXTRL_RESTRICT data_buffer, Iter offset_begin, Iter offset_end,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size )
    {
        return BufferArrayObj::AddToBuffer( *buffer.getCApiPtr(), data_buffer,
            offset_begin, offset_end, max_nelements, capacity, base_type_id,
                slot_size );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj::address_t const data_begin_addr,
        BufferArrayObj::address_t const offset_list_begin_addr,
        BufferArrayObj::size_type const num_elements,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size  )
    {
        return BufferArrayObj::AddToBuffer( &buffer, data_begin_addr,
            offset_list_begin_addr, num_elements, max_nelements, capacity,
                base_type_id, slot_size );
    }

    template< typename Iter >
    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj_add(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_BUFFER_DATAPTR_DEC const void *const SIXTRL_RESTRICT data_buffer,
        Iter offset_begin, Iter offset_end,
        BufferArrayObj::size_type const max_nelements,
        BufferArrayObj::size_type const capacity,
        BufferArrayObj::type_id_t const base_type_id,
        BufferArrayObj::size_type const slot_size )
    {
        return BufferArrayObj::AddToBuffer( &buffer, data_buffer, offset_begin,
            offset_end, max_nelements, capacity, base_type_id, slot_size );
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC SIXTRL_FN BufferArrayObj::ptr_stored_t BufferArrayObj_add_copy(
        BufferArrayObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj const& SIXTRL_RESTRICT_REF orig )
    {
        return BufferArrayObj::AddToBuffer( *buffer.getCApiPtr(),
            orig.getDataBeginAddress(), orig.getElementBeginAddress(),
            orig.getNumElements(), orig.getMaxNumElements(),
            orig.getCapacity(), orig.getBaseTypeId(), orig.getSlotSize() );
    }

    SIXTRL_INLINE BufferArrayObj::ptr_stored_t BufferArrayObj_add_copy(
        BufferArrayObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferArrayObj const& SIXTRL_RESTRICT_REF orig )
    {
        return BufferArrayObj::AddToBuffer( &buffer, orig.getDataBeginAddress(),
            orig.getElementBeginAddress(), orig.getNumElements(),
            orig.getMaxNumElements(), orig.getCapacity(), orig.getBaseTypeId(),
            orig.getSlotSize() );
    }
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_HPP__ */

/* end: sixtracklib/common/buffer/buffer_array_object.hpp */
