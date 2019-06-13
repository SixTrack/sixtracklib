#include "sixtracklib/common/internal/variant.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    Variant::Variant() :
    m_type( ::NS(VARIANT_TYPE_NONE) ), m_string()
    {

    }

    Variant::Variant(
        Variant::store_type_t const type, void const* begin,
        Variant::size_type const length  ) :
            m_type( ::NS(VARIANT_TYPE_NONE) ), m_string()
    {
        using size_t = Variant::size_type;

        if( ( begin != nullptr ) && ( length > size_t{ 0 } ) &&
            ( type != ::NS(VARIANT_TYPE_AUTO) ) &&
            ( type != ::NS(VARIANT_TYPE_NONE) ) &&
            ( type != ::NS(VARIANT_TYPE_UNDEFINED) ) )
        {
            void*  dest_addr   = nullptr;
            size_t dest_length = size_t{ 0 };

            switch( type )
            {
                case NS(VARIANT_TYPE_CHAR):
                {
                    dest_addr   = &( this->m_char );
                    dest_length = sizeof( char );
                    break;
                }

                case NS(VARIANT_TYPE_UCHAR):
                {
                    dest_addr   = &( this->m_uchar );
                    dest_length = sizeof( unsigned char );
                    break;
                }

                case NS(VARIANT_TYPE_INT16):
                {
                    dest_addr   = &( this->m_int16 );
                    dest_length = sizeof( int16_t );
                    break;
                }

                case NS(VARIANT_TYPE_UINT16):
                {
                    dest_addr   = &( this->m_uint16 );
                    dest_length = sizeof( uint16_t );
                    break;
                }

                case NS(VARIANT_TYPE_INT32):
                {
                    dest_addr   = &( this->m_int32 );
                    dest_length = sizeof( int32_t );
                    break;
                }

                case NS(VARIANT_TYPE_UINT32):
                {
                    dest_addr   = &( this->m_uint32 );
                    dest_length = sizeof( uint32_t );
                    break;
                }

                case NS(VARIANT_TYPE_INT64):
                {
                    dest_addr   = &( this->m_int64 );
                    dest_length = sizeof( int64_t );
                    break;
                }

                case NS(VARIANT_TYPE_UINT64):
                {
                    dest_addr   = &( this->m_uint64 );
                    dest_length = sizeof( uint64_t );
                    break;
                }

                case NS(VARIANT_TYPE_FLOAT32):
                {
                    dest_addr   = &( this->m_float32 );
                    dest_length = sizeof( float );
                    break;
                }

                case NS(VARIANT_TYPE_FLOAT64):
                {
                    dest_addr   = &( this->m_float64);
                    dest_length = sizeof( double );
                    break;
                }

                case NS(VARIANT_TYPE_BOOL):
                {
                    dest_addr   = &( this->m_bool );
                    dest_length = sizeof( bool );
                    break;
                }

                case NS(VARIANT_TYPE_VOID_PTR):
                {
                    dest_addr   = &( this->m_void_ptr );
                    dest_length = sizeof( void* );
                    break;
                }

                case NS(VARIANT_TYPE_CHAR_PTR):
                {
                    dest_addr   = &( this->m_char_ptr );
                    dest_length = sizeof( char* );
                    break;
                }

                case NS(VARIANT_TYPE_UCHAR_PTR):
                {
                    dest_addr   = &( this->m_uchar_ptr );
                    dest_length = sizeof( unsigned char* );
                    break;
                }

                case NS(VARIANT_TYPE_BIN_ARRAY):
                {
                    dest_addr   = &( this->m_bin_array );
                    dest_length = sizeof( Variant::bin_array_t );
                    break;
                }


                case NS(VARIANT_TYPE_MANAGED_BUFFER):
                {
                    dest_addr   = &( this->m_managed_buffer );
                    dest_length = sizeof( Variant::managed_buffer_t );
                    break;
                }

                default:
                {
                    if( type == ::NS(VARIANT_TYPE_STRING) )
                    {
                        char const* begin_str =
                            reinterpret_cast< char const* >( begin );

                        size_t const in_str_len = std::strlen( begin_str );

                        if( ( in_str_len < length ) && ( length > size_t{ 0 } ) &&
                            ( begin_str[ length - size_t{ 1 } ] == '\0' ) )
                        {
                            this->m_string.clear();
                            this->m_string.reserve( length );
                            this->m_string += begin_str;
                        }
                    }
                }
            };

            if( ( dest_addr != nullptr ) && ( dest_addr != begin ) &&
                ( dest_length > size_t{ 0 } ) && ( dest_length >= length ) )
            {
                std::memcpy( dest_addr, begin, length );
                this->m_type = type;
            }
        }
    }

    Variant::~Variant() SIXTRL_NOEXCEPT
    {
        if( this->storedType() == ::NS(VARIANT_TYPE_STRING) )
        {
            this->m_string.~string();
        }
    }

    bool Variant::isSet() const SIXTRL_NOEXCEPT
    {
        return ( this->storedType() != ::NS(VARIANT_TYPE_NONE) );
    }

    void Variant::clear() SIXTRL_NOEXCEPT
    {
        this->m_type = ::NS(VARIANT_TYPE_NONE);
    }

    Variant::store_type_t
    Variant::storedType() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( this->m_type != ::NS(VARIANT_TYPE_AUTO) );
        SIXTRL_ASSERT( this->m_type != ::NS(VARIANT_TYPE_UNDEFINED) );

        return this->m_type;
    }

    std::string const& Variant::getString() const
    {
        if( this->storedType() != ::NS(VARIANT_TYPE_STRING) )
        {
            throw std::runtime_error( "variant does not store a string!" );
        }

        return this->m_string;
    }

    char const* Variant::getCString() const
    {
        if( this->storedType() != ::NS(VARIANT_TYPE_STRING) )
        {
            throw std::runtime_error( "variant does not store a string!" );
        }

        return this->m_string.c_str();

    }

    void Variant::setString(
        std::string const& SIXTRL_RESTRICT_REF str_value )
    {
        this->setString( str_value.c_str() );
    }

    void Variant::setString( const char *const SIXTRL_RESTRICT cstr )
    {
        using size_t = Variant::size_type;

        if( ( cstr == nullptr ) || ( std::strlen( cstr ) <= size_t{ 1 } ) )
        {
            throw std::runtime_error(
                "illegal input C string for adding to variant" );
        }

        this->setString( cstr, std::strlen( cstr ) + size_t{ 1 } );
    }

    void Variant::setString( const char *const SIXTRL_RESTRICT cstr,
         Variant::size_type const max_str_length )
    {
        using size_t = Variant::size_type;

        if( ( cstr == nullptr ) || ( max_str_length < size_t{ 1 } ) ||
            ( std::strlen( cstr ) >= max_str_length ) )
        {
            throw std::runtime_error(
                "illegal input C string for adding to variant" );
        }

        if( !this->m_string.empty() )
        {
            this->m_string.clear();
        }

        this->m_string.reserve( max_str_length );
        this->m_string += cstr;
        this->m_type = ::NS(VARIANT_TYPE_STRING);

        return;
    }

    void Variant::setManagedBuffer(
        Variant::c_buffer_t* SIXTRL_RESTRICT c_buffer ) SIXTRL_NOEXCEPT
    {
        this->setManagedBuffer(
            ::NS(Buffer_get_data_begin)( c_buffer ),
            ::NS(Buffer_get_slot_size)( c_buffer ) );
    }

    void Variant::setManagedBuffer(
        Variant::buffer_t& SIXTRL_RESTRICT_REF buffer ) SIXTRL_RESTRICT
    {
        this->setManagedBuffer(
            buffer.dataBegin< unsigned char* >(), buffer.getSlotSize() );
    }

    void Variant::setManagedBuffer(
        unsigned char* SIXTRL_RESTRICT data_begin,
        Variant::size_type const slot_size )
    {
        this->set< Variant::managed_buffer_t >(
            managed_buffer_t{ data_begin, slot_size } );
    }
}

#endif /* defined( __cplusplus ) */

/* end: sixtracklib/common/internal/variant.cpp */
