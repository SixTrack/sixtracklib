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
    #include "sixtracklib/common/buffer/buffer_string_object.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    class BufferStringObj : public ::NS(BufferStringObj)
    {
        public:

        using c_api_t       = ::NS(BufferStringObj);
        using buffer_t      = SIXTRL_CXX_NAMESPACE::Buffer;
        using size_type     = buffer_t::size_type;
        using address_t     = buffer_t::address_t;
        using c_buffer_t    = ::NS(Buffer);
        using object_t      = buffer_t::object_t;
        using type_id_t     = buffer_t::type_id_t;
        using ptr_stored_t  = SIXTRL_BUFFER_DATAPTR_DEC BufferStringObj*;

        using ptr_const_stored_t =
            SIXTRL_BUFFER_DATAPTR_DEC BufferStringObj const*;

        using ptr_size_arg_t =  SIXTRL_BUFFER_ARGPTR_DEC size_type*;
        using ptr_c_api_t    =  SIXTRL_BUFFER_DATAPTR_DEC BufferStringObj*;
        using ptr_object_t   =  SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t*;

        using ptr_const_object_t =
            SIXTRL_BUFFER_OBJ_ARGPTR_DEC object_t const*;

        explicit SIXTRL_FN BufferStringObj(
            address_t const begin_addr = address_t{ 0 },
            size_type const length = size_type{ 0 },
            size_type const capacity = size_type{ 0 } ) SIXTRL_NOEXCEPT;

        SIXTRL_FN BufferStringObj( BufferStringObj const& other ) = default;
        SIXTRL_FN BufferStringObj( BufferStringObj&& other ) = default;

        SIXTRL_FN BufferStringObj& operator=(
            const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN BufferStringObj& operator=(
            std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT;

        SIXTRL_FN BufferStringObj& operator=(
            BufferStringObj const& rhs ) = default;

        SIXTRL_FN BufferStringObj& operator=(
            BufferStringObj&& rhs ) = default;

        SIXTRL_FN ~BufferStringObj() = default;

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
            buffer_t& SIXTRL_RESTRICT_REF buffer, size_type const max_length,
                ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_objects  = nullptr,
                ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_slots    = nullptr,
                ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_dataptrs = nullptr
            ) SIXTRL_NOEXCEPT;

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t CreateNewOnBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const max_length );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            address_t const begin_address, size_type const length,
            size_type const capacity );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT cstring,
            size_type const max_length = size_type{ 0 } );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer,
            std::string const& SIXTRL_RESTRICT_REF str,
            size_type const max_length = size_type{ 0 } );

        template< typename Iter >
        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            buffer_t& SIXTRL_RESTRICT_REF buffer, Iter begin, Iter end,
                size_type const max_length = size_type{ 0 } );

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t CreateNewOnBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            size_type const max_length );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            address_t const begin_address,
            size_type const length, size_type const capacity );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT cstring,
            size_type const max_length = size_type{ 0 } );

        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer,
            std::string const& SIXTRL_RESTRICT_REF str,
            size_type const max_length = size_type{ 0 } );

        template< typename Iter >
        SIXTRL_STATIC SIXTRL_FN ptr_stored_t AddToBuffer(
            c_buffer_t& SIXTRL_RESTRICT_REF buffer, Iter begin, Iter end,
                size_type const max_length = size_type{ 0 } );

        /* ----------------------------------------------------------------- */

        SIXTRL_FN type_id_t getTypeId() const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumDataPtrs(
            size_type const max_length = size_type{ 0 },
            size_type const slot_size  = buffer_t::DEFAULT_SLOT_SIZE
        ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getNumSlots(
                size_type const max_length = size_type{ 0 },
                size_type const slot_size  = buffer_t::DEFAULT_SLOT_SIZE
        ) const SIXTRL_NOEXCEPT;

        SIXTRL_FN c_api_t const* getCApiPtr() const SIXTRL_NOEXCEPT;
        SIXTRL_FN c_api_t* getCApiPtr() SIXTRL_NOEXCEPT;

        SIXTRL_FN void preset() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN address_t getBeginAddress() const SIXTRL_NOEXCEPT;

        SIXTRL_FN char const* getCString() const SIXTRL_NOEXCEPT;
        SIXTRL_FN char* getCString() SIXTRL_NOEXCEPT;
        SIXTRL_FN std::string getString() const SIXTRL_NOEXCEPT;

        SIXTRL_FN char const* begin() const SIXTRL_NOEXCEPT;
        SIXTRL_FN char* begin() SIXTRL_NOEXCEPT;

        SIXTRL_FN char const* end() const SIXTRL_NOEXCEPT;
        SIXTRL_FN char* end() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_FN size_type getLength()    const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type size()         const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getMaxLength() const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type getCapacity()  const SIXTRL_NOEXCEPT;
        SIXTRL_FN size_type capacity()     const SIXTRL_NOEXCEPT;

        SIXTRL_FN size_type getAvailableLength() const SIXTRL_NOEXCEPT;

        SIXTRL_FN void  clear() SIXTRL_NOEXCEPT;
        SIXTRL_FN void  syncLength()  SIXTRL_NOEXCEPT;

        SIXTRL_FN char* assign(
            const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN char* assign(
            std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_FN char* assign( Iter begin, Iter end ) SIXTRL_NOEXCEPT;

        SIXTRL_FN char* append(
            const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN char* append(
            std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_FN char* append( Iter begin, Iter end ) SIXTRL_NOEXCEPT;

        SIXTRL_FN char* operator+=(
            const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT;

        SIXTRL_FN char* operator+=(
            std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT;

    };

    template<> struct ObjectTypeTraits< ::NS(BufferStringObj) >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return ::NS(OBJECT_TYPE_CSTRING);
        }
    };

    template<> struct ObjectTypeTraits< SIXTRL_CXX_NAMESPACE::BufferStringObj >
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_CSTRING);
        }
    };

    BufferStringObj::ptr_stored_t BufferStringObj_new(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const max_length );

    BufferStringObj::ptr_stored_t BufferStringObj_new(
        SIXTRL_BUFFER_ARGPTR_DEC BufferStringObj::c_buffer_t&
            SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const max_length );

    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::address_t const begin_addr,
        BufferStringObj::size_type const length,
        BufferStringObj::size_type const capacity );

    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        const char *const SIXTRL_RESTRICT cstr,
        BufferStringObj::size_type const max_length =
            BufferStringObj::size_type{ 0 } );

    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        std::string const& SIXTRL_RESTRICT_REF str,
        BufferStringObj::size_type const max_length =
            BufferStringObj::size_type{ 0 } );

    template< typename Iter >
    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        Iter begin, Iter end, BufferStringObj::size_type const max_length =
            BufferStringObj::size_type{ 0 } );


    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::address_t const begin_addr,
        BufferStringObj::size_type const length,
        BufferStringObj::size_type const capacity );

    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        const char *const SIXTRL_RESTRICT cstr,
        BufferStringObj::size_type const max_length =
            BufferStringObj::size_type{ 0 } );

    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        std::string const& SIXTRL_RESTRICT_REF str,
        BufferStringObj::size_type const max_length =
            BufferStringObj::size_type{ 0 } );

    template< typename Iter >
    BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        Iter begin, Iter end, BufferStringObj::size_type const max_length =
            BufferStringObj::size_type{ 0 } );


    BufferStringObj::ptr_stored_t BufferStringObj_add_copy(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj const& SIXTRL_RESTRICT_REF orig );

    BufferStringObj::ptr_stored_t BufferStringObj_add_copy(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj const& SIXTRL_RESTRICT_REF orig );
}

/* ************************************************************************ */
/* *******  BufferStringObj inline and template member functions    ******* */
/* ************************************************************************ */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <algorithm>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE BufferStringObj::BufferStringObj(
        BufferStringObj::address_t const begin_addr,
        BufferStringObj::size_type const length,
        BufferStringObj::size_type const capacity ) SIXTRL_NOEXCEPT :
        ::NS(BufferStringObj)()
    {
        using size_t = BufferStringObj::size_type;

        size_t const requ_capacity =
            ( ( length != size_t{ 0 } ) || ( capacity != size_t{ 0 } ) )
                ? std::max( length + size_t{ 1 }, capacity ) : size_t{ 0 };

        BufferStringObj::c_api_t* ptr = this->getCApiPtr();
        ::NS(BufferStringObj_set_begin_addr)( ptr, begin_addr );
        ::NS(BufferStringObj_set_length)( ptr, length );
        ::NS(BufferStringObj_set_capacity)( ptr, requ_capacity );
    }

    SIXTRL_INLINE BufferStringObj& BufferStringObj::operator=(
        const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT
    {
        ::NS(BufferStringObj_assign_cstring)( this->getCApiPtr(), cstr );
        return *this;
    }

    SIXTRL_INLINE BufferStringObj& BufferStringObj::operator=(
        std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT
    {
        ::NS(BufferStringObj_assign_cstring)(
            this->getCApiPtr(), str.c_str() );

        return *this;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE BufferStringObj::ptr_const_stored_t
    BufferStringObj::FromBuffer(
        BufferStringObj::buffer_t const& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        return BufferStringObj::FromBufferObject( buffer[ index ] );
    }

    SIXTRL_INLINE BufferStringObj::ptr_const_stored_t
    BufferStringObj::FromBuffer( SIXTRL_BUFFER_ARGPTR_DEC const
            BufferStringObj::c_buffer_t *const SIXTRL_RESTRICT buffer,
        BufferStringObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        return BufferStringObj::FromBufferObject(
            ::NS(Buffer_get_const_object)( buffer, index ) );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::FromBuffer(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        using _this_t = BufferStringObj;

        _this_t::buffer_t const& const_buffer = buffer;
        return const_cast< _this_t::ptr_stored_t >(
            _this_t::FromBuffer( const_buffer, index ) );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::FromBuffer(
        SIXTRL_BUFFER_ARGPTR_DEC BufferStringObj::c_buffer_t*
            SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const index ) SIXTRL_NOEXCEPT
    {
        using _this_t = BufferStringObj;

        _this_t::c_buffer_t const* const_buffer = buffer;
        return const_cast< _this_t::ptr_stored_t >(
            _this_t::FromBuffer( const_buffer, index ) );
    }

    SIXTRL_INLINE BufferStringObj::ptr_const_stored_t
    BufferStringObj::FromBufferObject( BufferStringObj::ptr_const_object_t
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
    {
        using str_obj_t = SIXTRL_CXX_NAMESPACE::BufferStringObj;
        using ptr_t = str_obj_t::ptr_const_stored_t;

        ptr_t ptr_str_obj = nullptr;

        if( ( obj != nullptr ) &&
            ( ::NS(Object_get_type_id)( obj ) == ::NS(OBJECT_TYPE_CSTRING) ) &&
            ( ::NS(Object_get_size)( obj ) >= sizeof( str_obj_t ) ) )
        {
            ptr_str_obj = reinterpret_cast< ptr_t >( static_cast< uintptr_t >(
                ::NS(Object_get_begin_addr)( obj ) ) );
        }

        return ptr_str_obj;
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t
    BufferStringObj::FromBufferObject( BufferStringObj::ptr_object_t
        SIXTRL_RESTRICT obj ) SIXTRL_NOEXCEPT
    {
        BufferStringObj::ptr_const_object_t cobj = obj;

        return const_cast< BufferStringObj::ptr_stored_t >(
            BufferStringObj::FromBufferObject( cobj ) );
    }

    SIXTRL_INLINE bool BufferStringObj::CanAddToBuffer(
            BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
            BufferStringObj::size_type const max_length,
            BufferStringObj::ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_objects,
            BufferStringObj::ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_slots,
            BufferStringObj::ptr_size_arg_t SIXTRL_RESTRICT ptr_requ_dataptrs
        ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_can_be_added)( buffer.getCApiPtr(),
            max_length, ptr_requ_objects, ptr_requ_slots, ptr_requ_dataptrs );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t
    BufferStringObj::CreateNewOnBuffer(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::CreateNewOnBuffer(
            *buffer.getCApiPtr(), max_length );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::address_t const begin_address,
        BufferStringObj::size_type const length,
        BufferStringObj::size_type const capacity )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), begin_address, length, capacity );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT cstring,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), cstring, max_length );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        std::string const& SIXTRL_RESTRICT_REF str,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), str.c_str(), max_length );
    }

    template< typename Iter >
    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        Iter begin, Iter end,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), begin, end, max_length );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t
    BufferStringObj::CreateNewOnBuffer(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const max_length )
    {
        return reinterpret_cast< BufferStringObj::ptr_stored_t >(
            ::NS(BufferStringObj_new)( &buffer, max_length ) );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::address_t const begin_address,
        BufferStringObj::size_type const length,
        BufferStringObj::size_type const capacity )
    {
        using ptr_t = BufferStringObj::ptr_stored_t;
        return reinterpret_cast< ptr_t >( ::NS(BufferStringObj_add_detailed)(
            &buffer, begin_address, length, capacity ) );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        SIXTRL_ARGPTR_DEC const char *const SIXTRL_RESTRICT cstring,
        BufferStringObj::size_type const max_length )
    {
        using size_t = BufferStringObj::size_type;
        using addr_t = BufferStringObj::address_t;
        using ptr_t  = BufferStringObj::ptr_stored_t;

        size_t const length =
            ::NS(BufferStringObj_get_cstring_length)( cstring );

        addr_t const begin_addr = static_cast< ::NS(buffer_addr_t) >(
            reinterpret_cast< uintptr_t >( cstring ) );

        size_t const requ_max_length = std::max( length, max_length );

        return reinterpret_cast< ptr_t >( ::NS(BufferStringObj_add_detailed)(
            &buffer, begin_addr, length, requ_max_length + size_t{ 1 } ) );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        std::string const& SIXTRL_RESTRICT_REF str,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer( buffer, str.c_str(), max_length );
    }

    template< typename Iter >
    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj::AddToBuffer(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        Iter begin, Iter end, BufferStringObj::size_type const max_length )
    {
        using diff_t = ::ptrdiff_t;
        using size_t = BufferStringObj::size_type;

        BufferStringObj::ptr_stored_t added_str_obj = nullptr;

        diff_t const temp_len = std::distance( begin, end );

        if( temp_len >= diff_t{ 0 } )
        {
            size_t const length = static_cast< size_t >( temp_len );
            size_t const requ_max_length = std::max( length, max_length );

            added_str_obj = BufferStringObj::CreateNewOnBuffer(
                buffer,  requ_max_length );

            if( ( length > size_t{ 0 } ) && ( added_str_obj != nullptr ) )
            {
                char* ret = added_str_obj->assign( begin, end );
                SIXTRL_ASSERT( ret != nullptr );
                SIXTRL_ASSERT( added_str_obj->getLength() == length );
                ( void )ret;
            }
        }

        return added_str_obj;
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE BufferStringObj::type_id_t
    BufferStringObj::getTypeId() const SIXTRL_NOEXCEPT
    {
        return ::NS(OBJECT_TYPE_CSTRING);
    }

    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::getNumDataPtrs(
        BufferStringObj::size_type const max_length,
        BufferStringObj::size_type const slot_size ) const SIXTRL_NOEXCEPT
    {
        return
            ::NS(BufferStringObj_get_required_num_dataptrs_on_managed_buffer)(
                max_length, slot_size );
    }

    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::getNumSlots(
        BufferStringObj::size_type const max_length,
        BufferStringObj::size_type const slot_size ) const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_required_num_slots_on_managed_buffer)(
            max_length, slot_size );
    }

    SIXTRL_INLINE BufferStringObj::c_api_t const*
    BufferStringObj::getCApiPtr() const SIXTRL_NOEXCEPT
    {
        return reinterpret_cast< BufferStringObj::c_api_t const* >( this );
    }

    SIXTRL_INLINE BufferStringObj::c_api_t*
    BufferStringObj::getCApiPtr() SIXTRL_NOEXCEPT
    {
        return const_cast< BufferStringObj::c_api_t* >( static_cast<
            BufferStringObj const& >( *this ).getCApiPtr() );
    }

    SIXTRL_INLINE void BufferStringObj::preset() SIXTRL_NOEXCEPT
    {
        ::NS(BufferStringObj_preset)( this->getCApiPtr() );
    }


    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE char const*
    BufferStringObj::getCString() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_const_string)( this->getCApiPtr() );
    }

    SIXTRL_INLINE char* BufferStringObj::getCString() SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_string)( this->getCApiPtr() );
    }

    SIXTRL_INLINE std::string
    BufferStringObj::getString() const SIXTRL_NOEXCEPT
    {
        char const* ptr_cstring = this->getCString();
        return ( ptr_cstring != nullptr )
            ? std::string{ ptr_cstring } : std::string{};
    }

    SIXTRL_INLINE char const* BufferStringObj::begin() const SIXTRL_NOEXCEPT
    {
        return this->getCString();
    }

    SIXTRL_INLINE char* BufferStringObj::begin() SIXTRL_NOEXCEPT
    {
        return this->getCString();
    }

    SIXTRL_INLINE char const* BufferStringObj::end() const SIXTRL_NOEXCEPT
    {
        char const* end_ptr = this->begin();

        if( end_ptr != nullptr )
        {
            std::advance( end_ptr, this->getLength() );
        }

        return end_ptr;
    }

    SIXTRL_INLINE char* BufferStringObj::end() SIXTRL_NOEXCEPT
    {
        return const_cast< char* >( static_cast< BufferStringObj const& >(
            *this ).end() );
    }

    /* ----------------------------------------------------------------- */

    SIXTRL_INLINE BufferStringObj::address_t
    BufferStringObj::getBeginAddress() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_begin_addr)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::getLength() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_length)( this->getCApiPtr() );
    }


    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::size() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::getMaxLength() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_max_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::getCapacity()  const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_capacity)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::capacity() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_capacity)( this->getCApiPtr() );
    }

    SIXTRL_INLINE BufferStringObj::size_type
    BufferStringObj::getAvailableLength() const SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_get_available_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void BufferStringObj::clear() SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_clear)( this->getCApiPtr() );
    }

    SIXTRL_INLINE void BufferStringObj::syncLength()  SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_sync_length)( this->getCApiPtr() );
    }

    SIXTRL_INLINE char* BufferStringObj::assign(
        const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_assign_cstring)(
            this->getCApiPtr(), cstr );
    }

    SIXTRL_INLINE char* BufferStringObj::assign(
        std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_assign_cstring)(
            this->getCApiPtr(), str.c_str() );
    }

    template< typename Iter > SIXTRL_INLINE char* BufferStringObj::assign(
        Iter begin, Iter end ) SIXTRL_NOEXCEPT
    {
        using buf_size_t = ::NS(buffer_size_t);
        using diff_t     = ::ptrdiff_t;

        char* assigned_str = nullptr;
        ptrdiff_t const temp_len = std::distance( begin, end );

        if( temp_len >= diff_t{ 0 } )
        {
            buf_size_t const len = static_cast< buf_size_t >( temp_len );
            buf_size_t const capacity = this->getCapacity();

            assigned_str = this->begin();

            if( ( len > buf_size_t{ 0 } ) && ( assigned_str != nullptr ) )
            {
                std::copy( begin, end, assigned_str );
                std::fill( assigned_str + len, assigned_str + capacity, '\0' );

                ::NS(BufferStringObj_set_length)( this->getCApiPtr(), len );
             }
             else if( assigned_str != nullptr )
             {
                 this->clear();
             }
        }

        return assigned_str;
    }

    SIXTRL_INLINE char* BufferStringObj::append(
        const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_append_cstring)(
            this->getCApiPtr(), cstr );
    }

    SIXTRL_INLINE char* BufferStringObj::append(
        std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_append_cstring)(
            this->getCApiPtr(), str.c_str() );
    }

    template< typename Iter >
    SIXTRL_INLINE char* BufferStringObj::append(
        Iter begin, Iter end ) SIXTRL_NOEXCEPT
    {
        using buf_size_t  = ::NS(buffer_size_t);
        using diff_t      = ::ptrdiff_t;

        char* result_cstr = nullptr;
        ptrdiff_t const temp_len = std::distance( begin, end );

        if( temp_len > diff_t{ 0 } )
        {
            buf_size_t const current_length = this->getLength();
            buf_size_t const len = static_cast< buf_size_t >( temp_len );
            buf_size_t const avail_len = this->getAvailableLength();

            if( len <= avail_len )
            {
                char* ptr_dest = this->begin();
                std::advance( ptr_dest, current_length );
                std::copy( begin, end, ptr_dest );
                SIXTRL_ASSERT( *( ptr_dest + len ) == '\0' );

                ::NS(BufferStringObj_set_length)(
                    this->getCApiPtr(), len + current_length );

                SIXTRL_ASSERT( this->getAvailableLength() + len == avail_len );
                result_cstr = this->begin();;
             }
        }

        return result_cstr;
    }

    SIXTRL_INLINE char* BufferStringObj::operator+=(
        const char *const SIXTRL_RESTRICT cstr ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_append_cstring)(
            this->getCApiPtr(), cstr );
    }

    SIXTRL_INLINE char* BufferStringObj::operator+=(
        std::string const& SIXTRL_RESTRICT_REF str ) SIXTRL_NOEXCEPT
    {
        return ::NS(BufferStringObj_append_cstring)(
            this->getCApiPtr(), str.c_str() );
    }
}

/* ************************************************************************ */
/* *******  BufferStringObj free standing functions and methods     ******* */
/* ************************************************************************ */

namespace SIXTRL_CXX_NAMESPACE
{
    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_new(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::CreateNewOnBuffer(
            *buffer.getCApiPtr(), max_length );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_new(
        SIXTRL_BUFFER_ARGPTR_DEC BufferStringObj::c_buffer_t&
            SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::CreateNewOnBuffer( buffer, max_length );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::address_t const begin_addr,
        BufferStringObj::size_type const length,
        BufferStringObj::size_type const capacity )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), begin_addr, length, capacity );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        const char *const SIXTRL_RESTRICT cstr,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), cstr, max_length );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        std::string const& SIXTRL_RESTRICT_REF str,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), str.c_str(), max_length );
    }

    template< typename Iter >
    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        Iter begin, Iter end, BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer(
            *buffer.getCApiPtr(), begin, end, max_length );
    }


    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj::address_t const begin_addr,
        BufferStringObj::size_type const length,
        BufferStringObj::size_type const capacity )
    {
        return BufferStringObj::AddToBuffer(
            buffer, begin_addr, length, capacity );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        const char *const SIXTRL_RESTRICT cstr,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer( buffer, cstr, max_length );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        std::string const& SIXTRL_RESTRICT_REF str,
        BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer( buffer, str, max_length );
    }

    template< typename Iter >
    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        Iter begin, Iter end, BufferStringObj::size_type const max_length )
    {
        return BufferStringObj::AddToBuffer( buffer, begin, end, max_length );
    }


    SIXTRL_INLINE BufferStringObj::ptr_stored_t BufferStringObj_add_copy(
        BufferStringObj::buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj const& SIXTRL_RESTRICT_REF orig )
    {
        return BufferStringObj::AddToBuffer( *buffer.getCApiPtr(),
            orig.getBeginAddress(), orig.getLength(), orig.getCapacity() );
    }

    SIXTRL_INLINE BufferStringObj::ptr_stored_t BeamMonitor_add_copy(
        BufferStringObj::c_buffer_t& SIXTRL_RESTRICT_REF buffer,
        BufferStringObj const& SIXTRL_RESTRICT_REF orig )
    {
        return BufferStringObj::AddToBuffer( buffer,
            orig.getBeginAddress(), orig.getLength(), orig.getCapacity() );
    }
}

#endif /* C++, host */

#endif /* SIXTRACKLIB_COMMON_BUFFER_BUFFER_STRING_OBJECT_HPP__ */

/*end: sixtracklib/common/buffer/buffer_string_object.hpp */