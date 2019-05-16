#include "sixtracklib/common/control/argument_base.hpp"

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

#include "sixtracklib/common/control/controller_base.h"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    ArgumentBase::status_t ArgumentBase::send()
    {
        ArgumentBase::status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;

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
        using status_t = ArgumentBase::status_t;
        using size_t = ArgumentBase::size_type;
        using ptr_base_controller_t = ArgumentBase::ptr_base_controller_t;

        status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ptr_controller != nullptr )
        {
            success = ArgumentBase::STATUS_SUCCESS;

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
                        success = ArgumentBase::STATUS_GENERAL_FAILURE;
                    }
                }

                SIXTRL_ASSERT( requ_capacity <= this->capacity() );
            }

            if( ( success == ArgumentBase::STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesCObjectsCxxBuffer() ) ||
                  ( !this->usesCObjectsBuffer() ) ) )
            {
                this->doSetBufferRef( buffer );
            }

            if( success == ArgumentBase::STATUS_SUCCESS )
            {
                success = ptr_controller->send( this, buffer );

                if( success == ArgumentBase::STATUS_SUCCESS )
                {
                    this->doSetArgSize( buffer.size() );
                }
            }

            SIXTRL_ASSERT( ( success != ArgumentBase::STATUS_SUCCESS ) ||
                           ( buffer.size() == this->size() ) );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::send(
        const ArgumentBase::c_buffer_t *const SIXTRL_RESTRICT buffer )
    {
        using status_t = ArgumentBase::status_t;
        using size_t = ArgumentBase::size_type;
        using ptr_base_controller_t = ArgumentBase::ptr_base_controller_t;

        status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ptr_controller != nullptr )
        {
            success = ArgumentBase::STATUS_SUCCESS;

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
                        success = ArgumentBase::STATUS_GENERAL_FAILURE;
                    }
                }

                SIXTRL_ASSERT( requ_capacity <= this->capacity() );
            }

            if( ( success == ArgumentBase::STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesCObjectsBuffer() ) ) )
            {
                this->doSetPtrCBuffer( buffer );
            }

            if( success == ArgumentBase::STATUS_SUCCESS )
            {
                success = ptr_controller->send( this, buffer );

                if( success == ArgumentBase::STATUS_SUCCESS )
                {
                    this->doSetArgSize( ::NS(Buffer_get_size)( buffer ) );
                }
            }

            SIXTRL_ASSERT(
                ( success != ArgumentBase::STATUS_SUCCESS ) ||
                ( ::NS(Buffer_get_size)( buffer ) == this->size() ) );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::send(
        void const* SIXTRL_RESTRICT raw_arg_begin,
        ArgumentBase::size_type const raw_arg_len )
    {
        using status_t = ArgumentBase::status_t;
        using size_t = ArgumentBase::size_type;
        using ptr_base_controller_t = ArgumentBase::ptr_base_controller_t;

        status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) && ( raw_arg_begin != nullptr ) &&
            ( raw_arg_len > size_t{ 0 } ) )
        {
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
                    updated_argument_buffer =
                        this->doReserveArgumentBuffer( raw_arg_len );
                }

                if( raw_arg_len <= this->capacity() )
                {
                    success = ArgumentBase::STATUS_SUCCESS;
                }
            }
            else
            {
                success = ArgumentBase::STATUS_SUCCESS;
            }

            if( ( success == ArgumentBase::STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesRawArgument() ) ) )
            {
                this->doSetPtrRawArgument( raw_arg_begin );
            }

            if( success == ArgumentBase::STATUS_SUCCESS )
            {
                SIXTRL_ASSERT( ( !this->usesCObjectsBuffer() ) &&
                               ( !this->usesCObjectsCxxBuffer() ) );

                success = ptr_controller->send(
                    this, raw_arg_begin, raw_arg_len );

                if( success == ArgumentBase::STATUS_SUCCESS )
                {
                    this->doSetArgSize( raw_arg_len );
                }
            }

            SIXTRL_ASSERT( ( success != ArgumentBase::STATUS_SUCCESS ) ||
                           ( this->size() == raw_arg_len ) );

        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::receive()
    {
        ArgumentBase::status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;

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
        using ptr_base_controller_t = ArgumentBase::ptr_base_controller_t;

        status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) && ( this->usesCObjectsBuffer() ) )
        {
            success = ptr_controller->receive( buffer, this );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::receive(
        ArgumentBase::c_buffer_t* SIXTRL_RESTRICT buf )
    {

        using status_t = ArgumentBase::status_t;
        using ptr_base_controller_t = ArgumentBase::ptr_base_controller_t;

        status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) && ( this->usesCObjectsBuffer() ) )
        {
            success = ptr_controller->receive( buf, this );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::receive(
        void* SIXTRL_RESTRICT raw_arg_begin,
        ArgumentBase::size_type const raw_arg_capacity )
    {
        using status_t = ArgumentBase::status_t;
        using ptr_base_controller_t = ArgumentBase::ptr_base_controller_t;

        status_t success = ArgumentBase::STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) &&
            ( raw_arg_capacity >= this->size() ) )
        {
            success = ptr_controller->receive(
                raw_arg_begin, this->size(), this );
        }

        return success;
    }

    ArgumentBase::status_t ArgumentBase::remap()
    {
        using status_t = ArgumentBase::status_t;
        using ptr_base_controller_t = ArgumentBase::ptr_base_controller_t;

        status_t status = ArgumentBase::STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) &&
            ( ( this->usesCObjectsBuffer() ) ||
              ( this->usesCObjectsCxxBuffer() ) ) &&
            ( ( this->ptrCObjectsBuffer() != nullptr ) ) )
        {
            status = ptr_controller->remap( this );
        }

        return status;
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

    ArgumentBase::size_type
    ArgumentBase::cobjectsBufferSlotSize() const SIXTRL_NOEXCEPT
    {
        return ( this->usesCObjectsBuffer() )
            ? ::NS(Buffer_get_slot_size)( this->ptrCObjectsBuffer() )
            : ArgumentBase::size_type{ 0 };
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

    ArgumentBase::ptr_base_controller_t
    ArgumentBase::ptrControllerBase() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_controller;
    }

    ArgumentBase::ptr_const_base_controller_t
    ArgumentBase::ptrControllerBase() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_controller;
    }

    ArgumentBase::ArgumentBase(
        ArgumentBase::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str,
        bool const needs_argument_buffer,
        ArgumentBase::ptr_base_controller_t SIXTRL_RESTRICT controller ) :
        st::ArchBase( arch_id, arch_str, config_str ),
        m_ptr_raw_arg_begin( nullptr ),
        m_ptr_cobj_cxx_buffer( nullptr ),
        m_ptr_cobj_c99_buffer( nullptr ),
        m_ptr_base_controller( controller ),
        m_arg_size( ArgumentBase::size_type{ 0 } ),
        m_arg_capacity( ArgumentBase::size_type{ 0 } ),
        m_needs_arg_buffer( needs_argument_buffer ),
        m_has_arg_buffer( false )
    {

    }

    /* --------------------------------------------------------------------- */

    bool ArgumentBase::doReserveArgumentBuffer(
        ArgumentBase::size_type const required_arg_buffer_capacity )
    {
        return ( this->capacity() >= required_arg_buffer_capacity );
    }

    /* ----------------------------------------------------------------- */

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

    void ArgumentBase::doSetPtrControllerBase(
        ArgumentBase::ptr_base_controller_t SIXTRL_RESTRICT ctrl
    ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_base_controller = ctrl;
    }

    void ArgumentBase::doSetBufferRef( ArgumentBase::buffer_t const&
        SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        using c_buffer_t = ArgumentBase::c_buffer_t;
        using buffer_t   = ArgumentBase::buffer_t;

        c_buffer_t* _ptr_cobj_c99_buffer =
            const_cast< c_buffer_t* >( buffer.getCApiPtr() );

        if( _ptr_cobj_c99_buffer != this->m_ptr_cobj_c99_buffer )
        {
            this->m_ptr_cobj_c99_buffer = _ptr_cobj_c99_buffer;
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

        c_buffer_t* non_const_ptr = const_cast< c_buffer_t* >( ptr_c_buffer );

        if( this->m_ptr_cobj_cxx_buffer != nullptr )
        {
            SIXTRL_ASSERT( this->m_ptr_cobj_cxx_buffer->getCApiPtr() ==
                           this->m_ptr_cobj_c99_buffer );

            if( this->m_ptr_cobj_cxx_buffer->getCApiPtr() != non_const_ptr )
            {
                this->doResetPtrCxxBuffer();
            }
        }

        SIXTRL_ASSERT( ( this->m_ptr_cobj_cxx_buffer == nullptr ) ||
            ( ( this->m_ptr_cobj_cxx_buffer->getCApiPtr() == non_const_ptr ) &&
              ( this->m_ptr_cobj_c99_buffer ==
                this->m_ptr_cobj_cxx_buffer->getCApiPtr() ) ) );

        this->m_ptr_cobj_c99_buffer = non_const_ptr;
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

/* end: sixtracklib/common/control/argument_base.cpp */
