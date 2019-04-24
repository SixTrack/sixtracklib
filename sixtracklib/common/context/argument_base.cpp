#include "sixtracklib/common/context/argument_base.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/buffer.hpp"

#include "sixtracklib/common/context/context_base.h"

namespace SIXTRL_CXX_NAMESPACE
{
    namespace st = SIXTRL_CXX_NAMESPACE;

    ArgumentBase::status_t ArgumentBase::send()
    {
        ArgumentBase::status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;

        if( this->usesCObjectsCxxBuffer() )
        {
            success = this->send( this->cobjectsCxxBuffer() );
        }
        else if( this->usesCObjectsBuffer() )
        {
            success = this->send( this->ptrCObjectsBuffer() );
        }
        else if( this->usesRawArgument() )
        {
            success = this->send( this->ptrRawArgument(), this->size() );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::send(
        ArgumentBase::buffer_t const& SIXTRL_RESTRICT_REF buffer )
    {
        using status_t           = ArgumentBase::status_t;
        using size_t             = ArgumentBase::size_type;
        using ptr_base_context_t = ArgumentBase::ptr_base_context_t;

        status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;
        ptr_base_context_t ptr_context = this->ptrBaseContext();

        if( ptr_context != nullptr )
        {
            success = st::CONTEXT_STATUS_SUCCESS;

            if( this->usesRawArgument() )
            {
                this->doSetPtrRawArgument( nullptr );
            }

            bool updated_argument_buffer = false;

            if( this->requiresArgumentBuffer() )
            {
                size_t const requ_capacity = buffer.size();

                if( requ_capacity > this->capacity() )
                {
                    bool const reserved_success =
                        this->doReserveArgumentBuffer( requ_capacity );

                    updated_argument_buffer = reserved_success;

                    if( !reserved_success )
                    {
                        success = st::CONTEXT_STATUS_GENERAL_FAILURE;
                    }
                }

                SIXTRL_ASSERT( requ_capacity <= this->capacity() );
            }

            if( ( success == st::CONTEXT_STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesCObjectsCxxBuffer() ) ||
                  ( !this->usesCObjectsBuffer() ) ) )
            {
                this->doSetBufferRef( buffer );
            }

            if( success == st::CONTEXT_STATUS_SUCCESS )
            {
                success = ptr_context->send( this, buffer );

                if( success == st::CONTEXT_STATUS_SUCCESS )
                {
                    this->doSetArgSize( buffer.size() );
                }
            }

            SIXTRL_ASSERT( ( success != st::CONTEXT_STATUS_SUCCESS ) ||
                           ( buffer.size() == this->size() ) );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::send(
        const ArgumentBase::c_buffer_t *const SIXTRL_RESTRICT buffer )
    {
        using status_t           = ArgumentBase::status_t;
        using size_t             = ArgumentBase::size_type;
        using ptr_base_context_t = ArgumentBase::ptr_base_context_t;

        status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;
        ptr_base_context_t ptr_context = this->ptrBaseContext();

        if( ptr_context != nullptr )
        {
            success = st::CONTEXT_STATUS_SUCCESS;

            if( this->usesRawArgument() )
            {
                this->doSetPtrRawArgument( nullptr );
            }

            bool updated_argument_buffer = false;

            if( this->requiresArgumentBuffer() )
            {
                size_t const requ_capacity = ::NS(Buffer_get_size)( buffer );

                if( requ_capacity > this->capacity() )
                {
                    bool const reserved_success =
                        this->doReserveArgumentBuffer( requ_capacity );

                    updated_argument_buffer = reserved_success;

                    if( !reserved_success )
                    {
                        success = st::CONTEXT_STATUS_GENERAL_FAILURE;
                    }
                }

                SIXTRL_ASSERT( requ_capacity <= this->capacity() );
            }

            if( ( success == st::CONTEXT_STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesCObjectsBuffer() ) ) )
            {
                this->doSetPtrCBuffer( buffer );
            }

            if( success == st::CONTEXT_STATUS_SUCCESS )
            {
                success = ptr_context->send( this, buffer );

                if( success == st::CONTEXT_STATUS_SUCCESS )
                {
                    this->doSetArgSize( ::NS(Buffer_get_size)( buffer ) );
                }
            }

            SIXTRL_ASSERT(
                ( success != st::CONTEXT_STATUS_SUCCESS ) ||
                ( ::NS(Buffer_get_size)( buffer ) == this->size() ) );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::send(
        void const* SIXTRL_RESTRICT raw_arg_begin,
        ArgumentBase::size_type const raw_arg_len )
    {
        using status_t           = ArgumentBase::status_t;
        using size_t             = ArgumentBase::size_type;
        using ptr_base_context_t = ArgumentBase::ptr_base_context_t;

        status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;
        ptr_base_context_t ptr_context = this->ptrBaseContext();

        if( ( ptr_context != nullptr ) && ( raw_arg_begin != nullptr ) &&
            ( raw_arg_len > size_t{ 0 } ) )
        {
            success = true;

            bool updated_argument_buffer = false;

            if( this->usesCObjectsCxxBuffer() )
            {
                this->doResetPtrCxxBuffer();
            }
            else if( this->usesCObjectsBuffer() )
            {
                this->doSetPtrCBuffer( nullptr );
            }

            if( this->requiresArgumentBuffer() )
            {
                if( raw_arg_len > this->capacity() )
                {
                    bool const reserved_success =
                        this->doReserveArgumentBuffer( raw_arg_len );

                    updated_argument_buffer = reserved_success;

                    if( !reserved_success )
                    {
                        success = st::CONTEXT_STATUS_GENERAL_FAILURE;
                    }
                }

                SIXTRL_ASSERT( raw_arg_len <= this->capacity() );
            }

            if( ( success == st::CONTEXT_STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesRawArgument() ) ) )
            {
                this->doSetPtrRawArgument( raw_arg_begin );
            }

            if( success == st::CONTEXT_STATUS_SUCCESS )
            {
                SIXTRL_ASSERT( ( !this->usesCObjectsBuffer() ) &&
                               ( !this->usesCObjectsCxxBuffer() ) );

                success = ptr_context->send( this, raw_arg_begin, raw_arg_len );

                if( success == st::CONTEXT_STATUS_SUCCESS )
                {
                    this->doSetArgSize( raw_arg_len );
                }
            }

            SIXTRL_ASSERT( ( success != st::CONTEXT_STATUS_SUCCESS ) ||
                           ( this->size() == raw_arg_len ) );

        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::receive()
    {
        ArgumentBase::status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;

        if( this->usesCObjectsCxxBuffer() )
        {
            success = this->receive( this->cobjectsCxxBuffer() );
        }
        else if( this->usesCObjectsBuffer() )
        {
            success = this->receive( this->ptrCObjectsBuffer() );
        }
        else if( this->usesRawArgument() )
        {
            success = this->receive( this->ptrRawArgument(), this->size() );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::receive(
        ArgumentBase::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {

        using status_t = ArgumentBase::status_t;
        using ptr_base_context_t = ArgumentBase::ptr_base_context_t;

        status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;
        ptr_base_context_t ptr_context = this->ptrBaseContext();

        if( ( ptr_context != nullptr ) && ( this->usesCObjectsBuffer() ) )
        {
            success = ptr_context->receive( buffer, this );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::receive(
        ArgumentBase::c_buffer_t* SIXTRL_RESTRICT buf )
    {

        using status_t = ArgumentBase::status_t;
        using ptr_base_context_t = ArgumentBase::ptr_base_context_t;

        status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;
        ptr_base_context_t ptr_context = this->ptrBaseContext();

        if( ( ptr_context != nullptr ) && ( this->usesCObjectsBuffer() ) )
        {
            success = ptr_context->receive( buf, this );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::receive( void* SIXTRL_RESTRICT raw_arg_begin,
        ArgumentBase::size_type const raw_arg_capacity )
    {
        using status_t = ArgumentBase::status_t;
        using ptr_base_context_t = ArgumentBase::ptr_base_context_t;

        status_t success = st::CONTEXT_STATUS_GENERAL_FAILURE;
        ptr_base_context_t ptr_context = this->ptrBaseContext();

        if( ( ptr_context != nullptr ) &&
            ( raw_arg_capacity >= this->size() ) )
        {
            success = ptr_context->receive( raw_arg_begin, this->size(), this );
        }

        return success;
    }

    bool ArgumentBase::usesCObjectsCxxBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_cobj_cxx_buffer == nullptr ) ||
            ( ( this->m_ptr_cobj_c99_buffer != nullptr ) &&
              ( this->m_ptr_cobj_cxx_buffer->getCApiPtr() ==
                this->m_ptr_cobj_c99_buffer ) ) );

        return ( this->m_ptr_cobj_cxx_buffer != nullptr );
    }

    ArgumentBase::buffer_t*
    ArgumentBase::ptrCObjectsCxxBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cobj_cxx_buffer;
    }

    ArgumentBase::buffer_t& ArgumentBase::cobjectsCxxBuffer() const
    {
        if( this->m_ptr_cobj_cxx_buffer == nullptr )
        {
            throw std::runtime_error( "no C++ CObjects buffer available" );
        }

        return *this->m_ptr_cobj_cxx_buffer;
    }

    bool ArgumentBase::usesCObjectsBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_cobj_cxx_buffer == nullptr ) ||
            ( ( this->m_ptr_cobj_c99_buffer != nullptr ) &&
              ( this->m_ptr_cobj_cxx_buffer->getCApiPtr() ==
                this->m_ptr_cobj_c99_buffer ) ) );

        return ( this->m_ptr_cobj_c99_buffer != nullptr );
    }

    ArgumentBase::c_buffer_t*
    ArgumentBase::ptrCObjectsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cobj_c99_buffer;
    }

    bool ArgumentBase::usesRawArgument() const SIXTRL_NOEXCEPT
    {
        return ( this->m_ptr_raw_arg_begin != nullptr );
    }

    void* ArgumentBase::ptrRawArgument() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_raw_arg_begin;
    }

    ArgumentBase::size_type ArgumentBase::size() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_size;
    }

    ArgumentBase::size_type ArgumentBase::capacity() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_capacity;
    }

    bool ArgumentBase::hasArgumentBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_has_arg_buffer;
    }

    bool ArgumentBase::requiresArgumentBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_needs_arg_buffer;
    }

    ArgumentBase::ptr_base_context_t
    ArgumentBase::ptrBaseContext() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_context;
    }

    ArgumentBase::ptr_const_base_context_t
    ArgumentBase::ptrBaseContext() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_context;
    }

    ArgumentBase::type_id_t ArgumentBase::type() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id;
    }

    std::string const& ArgumentBase::typeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id_str;
    }

    char const* ArgumentBase::ptrTypeStr() const SIXTRL_NOEXCEPT
    {
        return this->m_type_id_str.c_str();
    }

    ArgumentBase::ArgumentBase(
        ArgumentBase::type_id_t const type_id,
        const char *const SIXTRL_RESTRICT type_id_str,
        bool const needs_argument_buffer,
        ArgumentBase::ptr_base_context_t SIXTRL_RESTRICT context ) :
        m_type_id_str(),
        m_ptr_raw_arg_begin( nullptr ),
        m_ptr_cobj_cxx_buffer( nullptr ),
        m_ptr_cobj_c99_buffer( nullptr ),
        m_ptr_base_context( context ),
        m_arg_size( ArgumentBase::size_type{ 0 } ),
        m_arg_capacity( ArgumentBase::size_type{ 0 } ),
        m_type_id( type_id ),
        m_needs_arg_buffer( needs_argument_buffer ),
        m_has_arg_buffer( false )
    {
        this->doSetTypeIdStr( type_id_str );
    }

    /* --------------------------------------------------------------------- */

    bool ArgumentBase::doReserveArgumentBuffer(
        ArgumentBase::size_type const required_arg_buffer_capacity )
    {
        return ( this->capacity() >= required_arg_buffer_capacity );
    }

    /* ----------------------------------------------------------------- */

    void ArgumentBase::doSetTypeId(
        ArgumentBase::type_id_t const type_id ) SIXTRL_NOEXCEPT
    {
        this->m_type_id = type_id;
    }

    void ArgumentBase::doSetTypeIdStr(
        const char *const SIXTRL_RESTRICT type_id_str ) SIXTRL_NOEXCEPT
    {
        if( ( type_id_str != nullptr ) &&
            ( std::strlen( type_id_str ) > std::size_t{ 0 } ) )
        {
            this->m_type_id_str = std::string{ type_id_str };
        }
        else
        {
            this->m_type_id_str.clear();
        }
    }

    void ArgumentBase::doSetArgSize(
        ArgumentBase::size_type const arg_size ) SIXTRL_NOEXCEPT
    {
        this->m_arg_size = arg_size;
    }

    void ArgumentBase::doSetArgCapacity(
        ArgumentBase::size_type const arg_capacity ) SIXTRL_NOEXCEPT
    {
        this->m_arg_capacity = arg_capacity;
    }

    void ArgumentBase::doSetPtrContext( ArgumentBase::ptr_base_context_t
        SIXTRL_RESTRICT ptr_context ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_base_context = ptr_context;
    }

    void ArgumentBase::doSetBufferRef( ArgumentBase::buffer_t const&
        SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        using c_buffer_t = ArgumentBase::c_buffer_t;
        using buffer_t   = ArgumentBase::buffer_t;

        c_buffer_t* new_ptr_cobj_c99_buffer = const_cast< c_buffer_t* >(
            buffer.getCApiPtr() );

        if( new_ptr_cobj_c99_buffer != this->m_ptr_cobj_c99_buffer )
        {
            this->m_ptr_cobj_c99_buffer = new_ptr_cobj_c99_buffer;
        }

        this->m_ptr_cobj_cxx_buffer = const_cast< buffer_t* >( &buffer );
    }

    void ArgumentBase::doResetPtrCxxBuffer() SIXTRL_NOEXCEPT
    {
        this->m_ptr_cobj_cxx_buffer =  nullptr;
    }

    void ArgumentBase::doSetPtrCBuffer(
        const ArgumentBase::c_buffer_t *const
        SIXTRL_RESTRICT ptr_c_buffer ) SIXTRL_NOEXCEPT
    {
        using c_buffer_t = ArgumentBase::c_buffer_t;

        c_buffer_t* non_const_new_ptr =
            const_cast< c_buffer_t* >( ptr_c_buffer );

        if( this->m_ptr_cobj_cxx_buffer != nullptr )
        {
            SIXTRL_ASSERT( this->m_ptr_cobj_cxx_buffer->getCApiPtr() ==
                           this->m_ptr_cobj_c99_buffer );

            if( this->m_ptr_cobj_cxx_buffer->getCApiPtr() !=
                non_const_new_ptr )
            {
                this->doResetPtrCxxBuffer();
            }
        }

        SIXTRL_ASSERT( ( this->m_ptr_cobj_cxx_buffer == nullptr ) ||
            ( ( this->m_ptr_cobj_cxx_buffer->getCApiPtr() ==
                non_const_new_ptr ) &&
              ( this->m_ptr_cobj_c99_buffer ==
                this->m_ptr_cobj_cxx_buffer->getCApiPtr() ) ) );

        this->m_ptr_cobj_c99_buffer = non_const_new_ptr;
    }

    void ArgumentBase::doSetPtrRawArgument(
        const void *const SIXTRL_RESTRICT raw_arg_begin ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_raw_arg_begin = const_cast< void* >( raw_arg_begin );
    }

    void ArgumentBase::doSetHasArgumentBufferFlag(
        bool const is_available ) SIXTRL_NOEXCEPT
    {
        this->m_has_arg_buffer = is_available;
    }

    void ArgumentBase::doSetNeedsArgumentBufferFlag(
        bool const needs_argument_buffer ) SIXTRL_NOEXCEPT
    {
        this->m_needs_arg_buffer = needs_argument_buffer;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/common/context/argument_base.cpp */
