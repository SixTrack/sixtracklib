#ifndef SITRACKLIB_COMMON_INTERNAL_VARIANT_HPP_
#define SITRACKLIB_COMMON_INTERNAL_VARIANT_HPP_

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus ) && !defined( __CUDA_ARCH__ )
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

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

typedef enum NS(variant_store_type_e)
{
    NS(VARIANT_TYPE_UNDEFINED)      =   0,
    NS(VARIANT_TYPE_CHAR)           =   1,
    NS(VARIANT_TYPE_UCHAR)          =   2,
    NS(VARIANT_TYPE_INT8)           =   1,
    NS(VARIANT_TYPE_UINT8)          =   2,
    NS(VARIANT_TYPE_INT16)          =   5,
    NS(VARIANT_TYPE_UINT16)         =   6,
    NS(VARIANT_TYPE_INT32)          =   7,
    NS(VARIANT_TYPE_UINT32)         =   8,
    NS(VARIANT_TYPE_INT64)          =   9,
    NS(VARIANT_TYPE_UINT64)         =  10,
    NS(VARIANT_TYPE_FLOAT32)        =  11,
    NS(VARIANT_TYPE_FLOAT64)        =  12,
    NS(VARIANT_TYPE_BOOL)           =  13,
    NS(VARIANT_TYPE_VOID_PTR)       =  64,
    NS(VARIANT_TYPE_CHAR_PTR)       =  65,
    NS(VARIANT_TYPE_UCHAR_PTR)      =  66,
    NS(VARIANT_TYPE_BIN_ARRAY)      = 128,
    NS(VARIANT_TYPE_CSTRING)        = 129,
    NS(VARIANT_TYPE_CBUFFER)        = 130,
    NS(VARIANT_TYPE_AUTO)           = 65534,
    NS(VARIANT_TYPE_NONE)           = 65535
}
NS(variant_store_type_t);

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    using variant_store_type_t = ::NS(variant_store_type_t);

    template< typename T >
    struct TVariantStoreTypeTraits
    {
        using value_type            = T;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_UNDEFINED);
        }
    };

    class Variant
    {
        public:

        using size_type = ::NS(buffer_size_t);
        using c_buffer_t   = ::NS(Buffer);
        using buffer_t     = SIXTRL_CXX_NAMESPACE::Buffer;
        using store_type_t = ::NS(variant_store_type_t);

        static size_type constexpr
            DEFAULT_ALIGNMENT = size_type{ 8u };

        SIXTRL_HOST_FN explicit Variant(
            size_type const alignment = DEFAULT_ALIGNMENT );

        SIXTRL_HOST_FN Variant( Variant const& other ) = default;
        SIXTRL_HOST_FN Variant( Variant&& other ) = default;

        SIXTRL_HOST_FN Variant& operator=( Variant const& rhs ) = default;
        SIXTRL_HOST_FN Variant& operator=( Variant&& rhs ) = default;

        SIXTRL_HOST_FN ~Variant() SIXTRL_NOEXCEPT;

        store_type_t storedType( size_type const ) const SIXTRL_NOEXCEPT;
        bool isSet() const SIXTRL_NOEXCEPT;

        size_type size()      const SIXTRL_NOEXCEPT;
        size_type capacity()  const SIXTRL_NOEXCEPT;
        size_type alignment() const SIXTRL_NOEXCEPT;
        void reserve( size_type const new_capacity );

        void clear() SIXTRL_NOEXCEPT;

        template< typename T > bool
        SIXTRL_HOST_FN isA() const SIXTRL_NOEXCEPT;

        template< typename T >
        SIXTRL_HOST_FN typename TVariantStoreTypeTraits<T>::value_type
        as() const;

        template< typename T >
        SIXTRL_HOST_FN typename TVariantStoreTypeTraits<T>::pointer
        asPtr() const;

        template< typename T >
        SIXTRL_HOST_FN typename TVariantStoreTypeTraits<T>::const_pointer
        asPtr() const;

        store_type_t setData( void const* SIXTRL_RESTRICT value_begin,
            size_type const value_size, store_type_t const type_hint );

        SIXTRL_HOST_FN store_type_t set( char const value );
        SIXTRL_HOST_FN store_type_t set( unsigned char const value );
        SIXTRL_HOST_FN store_type_t set( int16_t  const value );
        SIXTRL_HOST_FN store_type_t set( uint16_t const value );
        SIXTRL_HOST_FN store_type_t set( int32_t  const value );
        SIXTRL_HOST_FN store_type_t set( uint32_t const value );
        SIXTRL_HOST_FN store_type_t set( int64_t  const value );
        SIXTRL_HOST_FN store_type_t set( uint64_t const value );
        SIXTRL_HOST_FN store_type_t set( float    const value );
        SIXTRL_HOST_FN store_type_t set( double   const value );
        SIXTRL_HOST_FN store_type_t set( bool     const value );
        SIXTRL_HOST_FN store_type_t set( void* SIXTRL_RESTRICT ptr );
        SIXTRL_HOST_FN store_type_t set( char* SIXTRL_RESTRICT ptr );
        SIXTRL_HOST_FN store_type_t set( unsigned char* SIXTRL_RESTRICT ptr );
        SIXTRL_HOST_FN store_type_t set(
            const void *const SIXTRL_RESTRICT bin_array_begin,
            size_type const bin_array_length,
            store_type_t const type_hint = ::NS(VARIANT_TYPE_BIN_ARRAY) );

        SIXTRL_HOST_FN store_type_t set(
            std::string const& SIXTRL_RESTRICT_REF str );

        SIXTRL_HOST_FN store_type_t set(
            c_buffer_t* SIXTRL_RESTRICT ptr_buffer );

        SIXTRL_HOST_FN store_type_t set(
            buffer_t& SIXTRL_RESTRICT_REF buffer );

        private:

        using raw_store_t = std::vector< unsigned char >;

        SIXTRL_HOST_FN void* ptrValueBegin() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void const* ptrValueBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doDestructOldValue() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool doPrepareForNewValue(
            size_type const value_size ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool doConstructNewValue(
            store_type_t const type, void const* SIXTRL_RESTRICT value_begin,
            size_type const value_size );

        raw_store_t   m_data_store;
        void*         m_ptr_value_begin;
        size_type     m_value_size;
        size_type     m_alignment;
        size_type     m_offset;
        store_type_t  m_type;
    };
}

extern "C" {

typedef SIXTRL_CXX_NAMESPACE::Variant  NS(Variant);

}
#else /* defined( __cplusplus ) */

typedef void NS(Variant);

#endif /* defined( __cplusplus ) */

/* ========================================================================= */
/* Implementation of inline/template member functions, Traits specialization */
/* ========================================================================= */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename T > SIXTRL_INLINE bool Variant::isA() const
    {
        using type_t = Variant::store_type_t;
        type_t const expected = TVariantStoreTypeTraits<T>::GetStoreType();

        return ( ( this->isSet() ) && ( !this->m_data_store.empty() ) &&
                 ( this->storedType() == expected ) );
    }

    template< typename T >
    typename TVariantStoreTypeTraits< T >::value_type Variant::as()
    {
        using ptr_t = typename TVariantStoreTypeTraits< T >::pointer;
        ptr_t ptr = this->asPtr< T >();

        SIXTRL_ASSERT( ptr != nullptr );
        return *ptr;
    }

    template< typename T >
    typename TVariantStoreTypeTraits< T >::value_type Variant::as() const
    {
        using const_ptr = typename TVariantStoreTypeTraits< T >::const_pointer;
        const_ptr ptr = this->asPtr< T >();

        SIXTRL_ASSERT( ptr != nullptr );
        return *ptr;
    }

    SIXTRL_INLINE void* Variant::ptrValueBegin() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_value_begin;
    }

    SIXTRL_INLINE void const*
    Variant::ptrValueBegin() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_value_begin;
    }

    SIXTRL_INLINE void Variant::doDestructOldValue() SIXTRL_NOEXCEPT
    {
        if( this->isSet() )
        {
            switch( this->storedType() )
            {
                case ::NS(VARIANT_TYPE_CBUFFER):
                {
                    using c_buffer_t = Variant::c_buffer_t;
                    using ptr_t = TVariantStoreTypeTraits<
                        c_buffer_t >::pointer;

                    ptr_t ptr_buffer = this->asPtr< cbuffer_t >();
                    SIXTRL_ASSERT( ptr_buffer != nullptr );
                    ::NS(Buffer_free)( ptr_buffer );

                    break;
                }

                default:
                {
                }
            };

            this->m_type = ::NS(VARIANT_TYPE_NONE);
            this->m_value_size = size_type{ 0 };
            this->m_ptr_value_begin = nullptr;
        }

        return;
    }

    SIXTRL_INLINE bool Variant::doPrepareForNewValue(
        Variant::size_type const value_size,
        bool const fill_with_zeros ) SIXTRL_NOEXCEPT
    {
        bool success = false;

        using size_t    = Variant::size_type;
        using uintptr_t = std::uintptr_t;
        using raw_t     = unsigned char;

        if( ( !this->isSet() ) && ( this->m_alignment > size_t{ 0 } ) )
        {
            uintptr_t const begin_addr = reinterpret_cast< uintptr_t >(
                this->m_data_store.data() );

            SIXTRL_ASSERT( begin_addr != uintptr_t{ 0 } );

            uintptr_t align_remainder = begin_addr % this->m_alignment;

            size_t offset = size_t{ 0 };

            if( align_remainder > uintptr_t{ 0 } )
            {
                offset = this->m_alignment - align_remainder;
            }

            size_t required_capacity = offset + value_size;

            if( required_capacity >= this->m_data_store.size() )
            {
                required_capacity = this->m_alignment + value_size;

                this->m_data_store.clear();
                this->m_data_store.resize(
                    required_capacity, raw_t{ 0 } );

                begin_addr = reinterpret_cast< uintptr_t >(
                    this->m_data_store.data() );

                align_remainder = begin_addr % this->m_alignment;

                if( align_remainder > size_t{ 0 } )
                {
                    offset = this->m_alignment - align_remainder;
                }

                required_capacity = offset + value_size;
            }

            if( required_capacity <= this->m_data_store.size() )
            {
                raw_t* ptr_value_begin = this->m_data_store.data();
                std::advance( ptr_value_begin, offset );

                if( fill_with_zeros )
                {
                    raw_t* ptr_value_end = ptr_value_begin;
                    std::advance( ptr_value_end, value_size );
                    std::fill( ptr_value_begin, ptr_value_end, raw_t{ 0 } );
                }

                this->m_offset = offset;
                this->m_ptr_value_begin = reinterpret_cast< void* >(
                    ptr_value_begin );

                success = true;
            }
        }

        return success;
    }

    bool Variant::doConstructNewValue( Variant::store_type_t const type,
        void const* SIXTRL_RESTRICT value_begin,
        Variant::size_type const value_size )
    {
        bool success = false;

        using size_t = Variant::size_type;

        if( ( !this->isSet() ) && ( value_begin != nullptr ) &&
            ( value_size > size_t{ 0 } ) &&
            ( ( this->m_offset + value_size ) <= this->m_data_store.size() ) )
        {
            std::memcpy( this->m_ptr_value_begin, value_begin, value_size );
            this->m_value_size = value_size;
            this->m_type = type;

            switch( type )
            {
                case ::NS(VARIANT_TYPE_CBUFFER):
                {
                    using c_buffer_t = Variant::c_buffer_t;
                    c_buffer_t* ptr = this->asPtr< c_buffer_t >();
                    ::NS(Buffer_remap)( ptr );
                    success = !::NS(Buffer_needs_remapping)( ptr );

                    break;
                }

                default:
                {
                    success = true;
                }
            };
        }

        return success;
    }

    /* ******************************************************************** */

    template<> struct TVariantStoreTypeTraits< char >
    {
        using value_type            = char;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_CHAR);
        }
    };

    template<> struct TVariantStoreTypeTraits< unsigned char >
    {
        using value_type            = unsigned char;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_UCHAR);
        }
    };

    template<> struct TVariantStoreTypeTraits< int16_t >
    {
        using value_type            = int16_t;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_INT16);
        }
    };

    template<> struct TVariantStoreTypeTraits< uint16_t >
    {
        using value_type            = uint16_t;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_UINT16);
        }
    };

    template<> struct TVariantStoreTypeTraits< int32_t >
    {
        using value_type            = int32_t;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_INT32);
        }
    };

    template<> struct TVariantStoreTypeTraits< uint32_t >
    {
        using value_type            = uint32_t;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_UINT32);
        }
    };

    template<> struct TVariantStoreTypeTraits< int64_t >
    {
        using value_type            = int64_t;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_INT64);
        }
    };

    template<> struct TVariantStoreTypeTraits< uint64_t >
    {
        using value_type            = uint64_t;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_UINT64);
        }
    };

    template<> struct TVariantStoreTypeTraits< float >
    {
        using value_type            = float;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_FLOAT32);
        }
    };

    template<> struct TVariantStoreTypeTraits< double >
    {
        using value_type            = double;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_FLOAT64);
        }
    };

    template<> struct TVariantStoreTypeTraits< bool >
    {
        using value_type            = bool;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_BOOL);
        }
    };

    template<> struct TVariantStoreTypeTraits< void* >
    {
        using value_type            = void*;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_VOID_PTR);
        }
    };

    template<> struct TVariantStoreTypeTraits< char* >
    {
        using value_type            = char*;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_CHAR_PTR);
        }
    };

    template<> struct TVariantStoreTypeTraits< unsigned char* >
    {
        using value_type            = unsigned char*;
        using pointer               = value_type*;
        using const_pointer         = value_type const*;
        using param_type            = value_type;

        static variant_store_type_t GetStoreType() SIXTRL_NOEXCEPT
        {
            return ::NS(VARIANT_TYPE_UCHAR_PTR);
        }
    };
}

#endif /* defined( __cplusplus ) */


#endif /* SITRACKLIB_COMMON_INTERNAL_VARIANT_HPP_ */
/* end: sixtracklib/common/internal/variant.hpp */
