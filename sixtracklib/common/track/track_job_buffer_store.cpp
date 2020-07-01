#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/track/track_job_buffer_store.h"
#endif /* !defined SIXTRL_NO_INCLUDES */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <memory>
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
    #include "sixtracklib/common/buffer/assign_address_item.h"
    #include "sixtracklib/common/buffer.h"
    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    namespace
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using buf_store_t = st::TrackJobBufferStore;
        using st_size_t   = buf_store_t::size_type;
        using st_flags_t  = buf_store_t::flags_t;
    }

    buf_store_t::TrackJobBufferStore(
        st_size_t const buffer_capacity,
        st_flags_t const buffer_flags ) :
        m_ptr_cxx_buffer( nullptr ), m_ptr_c99_buffer( nullptr ),
        m_own_buffer( nullptr )
    {
        this->reset( buffer_capacity, buffer_flags );
    }

    buf_store_t::TrackJobBufferStore(
        buf_store_t::buffer_t* SIXTRL_RESTRICT cxx_buffer,
        bool const take_ownership ) :
        m_ptr_cxx_buffer( nullptr ), m_ptr_c99_buffer( nullptr ),
        m_own_buffer( nullptr )
    {
        this->reset( cxx_buffer, take_ownership );
    }

    buf_store_t::TrackJobBufferStore(
        buf_store_t::c_buffer_t* SIXTRL_RESTRICT c99_buffer,
        bool const take_ownership, bool const delete_ptr_after_move ) :
        m_ptr_cxx_buffer( nullptr ), m_ptr_c99_buffer( nullptr ),
        m_own_buffer( nullptr )
    {
        this->reset( c99_buffer, take_ownership, delete_ptr_after_move );
    }

    buf_store_t::TrackJobBufferStore(
        std::unique_ptr< buf_store_t::buffer_t >&& stored_ptr_buffer ) :
        m_ptr_cxx_buffer( nullptr ), m_ptr_c99_buffer( nullptr ),
        m_own_buffer( nullptr )
    {
        this->reset( std::move( stored_ptr_buffer ) );
    }

    buf_store_t::TrackJobBufferStore(
        buf_store_t::buffer_t&& cxx_buffer ) :
        m_ptr_cxx_buffer( nullptr ), m_ptr_c99_buffer( nullptr ),
        m_own_buffer( nullptr )
    {
        this->reset( std::move( cxx_buffer ) );
    }

    buf_store_t::TrackJobBufferStore(
        TrackJobBufferStore const& other ) :
        m_ptr_cxx_buffer( nullptr ), m_ptr_c99_buffer( nullptr ),
        m_own_buffer( nullptr )
    {
        if( other.owns_buffer() )
        {
            SIXTRL_ASSERT( other.m_own_buffer.get() != nullptr );
            this->m_own_buffer.reset(
                new st::Buffer( *other.m_own_buffer.get() ) );

            this->m_ptr_cxx_buffer = this->m_own_buffer.get();
            this->m_ptr_c99_buffer = this->m_own_buffer->getCApiPtr();
        }
        else
        {
            this->m_ptr_cxx_buffer = other.m_ptr_cxx_buffer;
            this->m_ptr_c99_buffer = other.m_ptr_c99_buffer;
        }
    }

    buf_store_t::TrackJobBufferStore(
        TrackJobBufferStore&& other ) SIXTRL_NOEXCEPT:
        m_ptr_cxx_buffer( std::move( other.m_ptr_cxx_buffer ) ),
        m_ptr_c99_buffer( std::move( other.m_ptr_c99_buffer ) ),
        m_own_buffer( std::move( other.m_own_buffer ) )
    {
        if( other.owns_buffer() )
        {
            SIXTRL_ASSERT( other.m_own_buffer.get() != nullptr );
            this->m_own_buffer.reset(
                new st::Buffer( *other.m_own_buffer.get() ) );

            this->m_ptr_cxx_buffer = this->m_own_buffer.get();
            this->m_ptr_c99_buffer = this->m_own_buffer->getCApiPtr();
        }
        else
        {
            this->m_ptr_cxx_buffer = other.m_ptr_cxx_buffer;
            this->m_ptr_c99_buffer = other.m_ptr_c99_buffer;
        }
    }

    TrackJobBufferStore& buf_store_t::operator=(
        TrackJobBufferStore const& rhs )
    {
        if( this != &rhs )
        {
            if( rhs.owns_buffer() )
            {
                SIXTRL_ASSERT( rhs.m_own_buffer.get() != nullptr );

                this->m_own_buffer.reset(
                    new st::Buffer( *rhs.m_own_buffer.get() ) );

                this->m_ptr_cxx_buffer = this->m_own_buffer.get();
                this->m_ptr_c99_buffer = this->m_own_buffer->getCApiPtr();
            }
            else
            {
                this->m_ptr_cxx_buffer = rhs.m_ptr_cxx_buffer;
                this->m_ptr_c99_buffer = rhs.m_ptr_c99_buffer;
            }
        }

        return *this;
    }

    TrackJobBufferStore&
    buf_store_t::operator=( TrackJobBufferStore&& rhs ) SIXTRL_NOEXCEPT
    {
        if( this != &rhs )
        {
            this->m_own_buffer = std::move( rhs.m_own_buffer );
            this->m_ptr_cxx_buffer = std::move( rhs.m_ptr_cxx_buffer );
            this->m_ptr_c99_buffer = std::move( rhs.m_ptr_c99_buffer );
        }

        return *this;
    }

    bool buf_store_t::active() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT(
            ( ( this->m_ptr_c99_buffer == nullptr ) &&
                ( this->m_ptr_cxx_buffer == nullptr ) &&
                ( this->m_own_buffer.get() == nullptr ) ) ||
            ( ( this->m_ptr_c99_buffer != nullptr ) &&
                ( ( this->m_ptr_cxx_buffer == nullptr ) ||
                ( this->m_ptr_cxx_buffer->getCApiPtr() ==
                    this->m_ptr_c99_buffer ) ) &&
                ( ( this->m_own_buffer == nullptr ) ||
                ( this->m_own_buffer.get() ==
                    this->m_ptr_cxx_buffer ) ) ) );

        return ( this->m_ptr_c99_buffer != nullptr );
    }

    bool buf_store_t::owns_buffer() const SIXTRL_NOEXCEPT
    {
        return ( ( this->active() ) &&
                 ( this->m_own_buffer.get() != nullptr ) );
    }

    buf_store_t::c_buffer_t const*
    buf_store_t::ptr_buffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_c99_buffer;
    }

    buf_store_t::c_buffer_t* buf_store_t::ptr_buffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_c99_buffer;
    }

    buf_store_t::buffer_t const*
    buf_store_t::ptr_cxx_buffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cxx_buffer;
    }

    buf_store_t::buffer_t* buf_store_t::ptr_cxx_buffer() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cxx_buffer;
    }

    void buf_store_t::clear() SIXTRL_NOEXCEPT
    {
        this->m_ptr_cxx_buffer = nullptr;
        this->m_ptr_c99_buffer = nullptr;
        this->m_own_buffer.reset( nullptr );
    }

    void buf_store_t::reset( st_size_t const buffer_capacity,
        st_flags_t const buffer_flags )
    {
        this->m_own_buffer.reset(
            new buf_store_t::buffer_t( buffer_capacity, buffer_flags ) );

        if( this->m_own_buffer.get() != nullptr )
        {
            this->m_ptr_cxx_buffer = this->m_own_buffer.get();
            this->m_ptr_c99_buffer = this->m_own_buffer->getCApiPtr();
        }
        else
        {
            this->m_ptr_cxx_buffer = nullptr;
            this->m_ptr_c99_buffer = nullptr;
        }
    }

    void buf_store_t::reset( buf_store_t::buffer_t* SIXTRL_RESTRICT cxx_buffer,
        bool const take_ownership )
    {
        if( ( cxx_buffer != nullptr ) &&
            ( this->m_ptr_cxx_buffer != cxx_buffer ) )
        {
            this->m_ptr_cxx_buffer = cxx_buffer;
            this->m_ptr_c99_buffer = cxx_buffer->getCApiPtr();

            if( take_ownership )
            {
                this->m_own_buffer.reset( cxx_buffer );
            }
            else if( this->m_own_buffer.get() != nullptr )
            {
                this->m_own_buffer.reset( nullptr );
            }
        }
    }

    void buf_store_t::reset( buf_store_t::c_buffer_t* SIXTRL_RESTRICT c99_buffer,
        bool const take_ownership, bool const delete_ptr_after_move )
    {
        using buffer_t = buf_store_t::buffer_t;

        if( ( c99_buffer != nullptr ) &&
            ( this->m_ptr_c99_buffer != c99_buffer ) )
        {
            if( take_ownership )
            {
                this->m_own_buffer =
                buffer_t::MAKE_FROM_CBUFFER_AND_TAKE_OWNERSHIP(
                    c99_buffer, delete_ptr_after_move );

                this->m_ptr_cxx_buffer = this->m_own_buffer.get();

                if( this->m_own_buffer.get() != nullptr )
                {
                    this->m_ptr_c99_buffer = this->m_own_buffer->getCApiPtr();
                }
            }
            else
            {
                this->m_ptr_c99_buffer = c99_buffer;
                this->m_ptr_cxx_buffer = nullptr;
                this->m_own_buffer.reset( nullptr );
            }
        }
    }

    void buf_store_t::reset(
        std::unique_ptr< buf_store_t::buffer_t >&& stored_ptr_buffer )
    {
        this->m_own_buffer = std::move( stored_ptr_buffer );
        this->m_ptr_cxx_buffer = this->m_own_buffer.get();

        this->m_ptr_c99_buffer = ( this->m_own_buffer.get() != nullptr )
            ? this->m_own_buffer->getCApiPtr() : nullptr;
    }

    void buf_store_t::reset( buf_store_t::buffer_t&& cxx_buffer )
    {
        if( &cxx_buffer != this->m_ptr_cxx_buffer )
        {
            this->m_own_buffer.reset(
                new buf_store_t::buffer_t( std::move( cxx_buffer ) ) );

            this->m_ptr_cxx_buffer = this->m_own_buffer.get();
            SIXTRL_ASSERT( this->m_own_buffer.get() != nullptr );
            this->m_ptr_c99_buffer = this->m_own_buffer->getCApiPtr();
        }
    }
}

#endif /* C++ */
