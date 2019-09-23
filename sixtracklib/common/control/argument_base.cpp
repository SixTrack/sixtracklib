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
    using _base_t = st::ArgumentBase;

    _base_t::status_t ArgumentBase::send(
        _base_t::perform_remap_flag_t const perform_remap_flag )
    {
        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->usesCObjectsCxxBuffer() )
        {
            success = this->send(
                this->cobjectsCxxBuffer(), perform_remap_flag );
        }
        else if( this->usesCObjectsBuffer() )
        {
            success = this->send(
                this->ptrCObjectsBuffer(), perform_remap_flag );
        }
        else if( this->usesRawArgument() )
        {
            success = this->send( this->ptrRawArgument(), this->size() );
        }

        return success;
    }

    _base_t::status_t ArgumentBase::send(
        _base_t::buffer_t const& SIXTRL_RESTRICT_REF buffer,
        _base_t::perform_remap_flag_t const perform_remap_flag )
    {
        using size_t = _base_t::size_type;
        using ptr_base_controller_t = _base_t::ptr_base_controller_t;

        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ptr_controller != nullptr )
        {
            success = st::ARCH_STATUS_SUCCESS;

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
                        success = st::ARCH_STATUS_GENERAL_FAILURE;
                    }
                }

                SIXTRL_ASSERT( requ_capacity <= this->capacity() );
            }

            if( ( success == st::ARCH_STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesCObjectsCxxBuffer() ) ||
                  ( !this->usesCObjectsBuffer() ) ) )
            {
                this->doSetBufferRef( buffer );
            }

            if( success == st::ARCH_STATUS_SUCCESS )
            {
                success = ptr_controller->send(
                    this, buffer, perform_remap_flag );

                if( success == st::ARCH_STATUS_SUCCESS )
                {
                    this->doSetArgSize( buffer.size() );
                }
            }

            SIXTRL_ASSERT( ( success != st::ARCH_STATUS_SUCCESS ) ||
                           ( buffer.size() == this->size() ) );
        }

        return success;
    }

    _base_t::status_t ArgumentBase::send(
        const _base_t::c_buffer_t *const SIXTRL_RESTRICT buffer,
        _base_t::perform_remap_flag_t const perform_remap_flag )
    {
        using size_t = _base_t::size_type;
        using ptr_base_controller_t = _base_t::ptr_base_controller_t;

        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ptr_controller != nullptr )
        {
            success = st::ARCH_STATUS_SUCCESS;

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
                        success = st::ARCH_STATUS_GENERAL_FAILURE;
                    }
                }

                SIXTRL_ASSERT( requ_capacity <= this->capacity() );
            }

            if( ( success == st::ARCH_STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesCObjectsBuffer() ) ) )
            {
                this->doSetPtrCBuffer( buffer );
            }

            if( success == st::ARCH_STATUS_SUCCESS )
            {
                success = ptr_controller->send(
                    this, buffer, perform_remap_flag );

                if( success == st::ARCH_STATUS_SUCCESS )
                {
                    this->doSetArgSize( ::NS(Buffer_get_size)( buffer ) );
                }
            }

            SIXTRL_ASSERT(
                ( success != st::ARCH_STATUS_SUCCESS ) ||
                ( ::NS(Buffer_get_size)( buffer ) == this->size() ) );
        }

        return success;
    }

    _base_t::status_t ArgumentBase::send(
        void const* SIXTRL_RESTRICT raw_arg_begin,
        _base_t::size_type const raw_arg_len )
    {
        using size_t = _base_t::size_type;
        using ptr_base_controller_t = _base_t::ptr_base_controller_t;

        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;
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
                    success = st::ARCH_STATUS_SUCCESS;
                }
            }
            else
            {
                success = st::ARCH_STATUS_SUCCESS;
            }

            if( ( success == st::ARCH_STATUS_SUCCESS ) &&
                ( ( updated_argument_buffer ) ||
                  ( !this->usesRawArgument() ) ) )
            {
                this->doSetPtrRawArgument( raw_arg_begin );
            }

            if( success == st::ARCH_STATUS_SUCCESS )
            {
                SIXTRL_ASSERT( ( !this->usesCObjectsBuffer() ) &&
                               ( !this->usesCObjectsCxxBuffer() ) );

                success = ptr_controller->send(
                    this, raw_arg_begin, raw_arg_len );

                if( success == st::ARCH_STATUS_SUCCESS )
                {
                    this->doSetArgSize( raw_arg_len );
                }
            }

            SIXTRL_ASSERT( ( success != st::ARCH_STATUS_SUCCESS ) ||
                           ( this->size() == raw_arg_len ) );

        }

        return success;
    }

    _base_t::status_t ArgumentBase::receive(
        _base_t::perform_remap_flag_t const perform_remap_flag )
    {
        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;

        if( this->usesCObjectsCxxBuffer() )
        {
            success = this->receive(
                this->cobjectsCxxBuffer(), perform_remap_flag );
        }
        else if( this->usesCObjectsBuffer() )
        {
            success = this->receive(
                this->ptrCObjectsBuffer(), perform_remap_flag );
        }
        else if( this->usesRawArgument() )
        {
            success = this->receive( this->ptrRawArgument(), this->size() );
        }

        return success;
    }

    _base_t::status_t ArgumentBase::receive(
        _base_t::buffer_t& SIXTRL_RESTRICT_REF buffer,
        _base_t::perform_remap_flag_t const perform_remap_flag )
    {
        using ptr_base_controller_t = _base_t::ptr_base_controller_t;
        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) && ( this->usesCObjectsBuffer() ) )
        {
            success = ptr_controller->receive(
                buffer, this, perform_remap_flag );
        }

        return success;
    }

    _base_t::status_t ArgumentBase::receive(
        _base_t::c_buffer_t* SIXTRL_RESTRICT buf,
        _base_t::perform_remap_flag_t const perform_remap_flag )
    {
        using ptr_base_controller_t = _base_t::ptr_base_controller_t;
        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) && ( this->usesCObjectsBuffer() ) )
        {
            success = ptr_controller->receive( buf, this, perform_remap_flag );
        }

        return success;
    }

    _base_t::status_t ArgumentBase::receive(
        void* SIXTRL_RESTRICT raw_arg_begin,
        _base_t::size_type const raw_arg_capacity )
    {
        using ptr_base_controller_t = _base_t::ptr_base_controller_t;
        _base_t::status_t success = st::ARCH_STATUS_GENERAL_FAILURE;
        ptr_base_controller_t ptr_controller = this->ptrControllerBase();

        if( ( ptr_controller != nullptr ) &&
            ( raw_arg_capacity >= this->size() ) )
        {
            success = ptr_controller->receive(
                raw_arg_begin, this->size(), this );
        }

        return success;
    }

    _base_t::status_t ArgumentBase::remap()
    {
        using ptr_base_controller_t = _base_t::ptr_base_controller_t;
        _base_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;
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

    _base_t::buffer_t* ArgumentBase::ptrCObjectsCxxBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cobj_cxx_buffer;
    }

    _base_t::buffer_t& ArgumentBase::cobjectsCxxBuffer() const
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

    _base_t::c_buffer_t* ArgumentBase::ptrCObjectsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cobj_c99_buffer;
    }

    _base_t::size_type ArgumentBase::cobjectsBufferSlotSize() const SIXTRL_NOEXCEPT
    {
        return ( this->usesCObjectsBuffer() )
            ? ::NS(Buffer_get_slot_size)( this->ptrCObjectsBuffer() )
            : _base_t::size_type{ 0 };
    }

    bool ArgumentBase::usesRawArgument() const SIXTRL_NOEXCEPT
    {
        return ( this->m_ptr_raw_arg_begin != nullptr );
    }

    void* ArgumentBase::ptrRawArgument() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_raw_arg_begin;
    }

    _base_t::size_type ArgumentBase::size() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_size;
    }

    _base_t::size_type ArgumentBase::capacity() const SIXTRL_NOEXCEPT
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

    _base_t::ptr_base_controller_t
    ArgumentBase::ptrControllerBase() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_controller;
    }

    _base_t::ptr_const_base_controller_t
    ArgumentBase::ptrControllerBase() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_controller;
    }

    ArgumentBase::ArgumentBase(
        _base_t::arch_id_t const arch_id,
        char const* SIXTRL_RESTRICT arch_str,
        char const* SIXTRL_RESTRICT config_str,
        bool const needs_argument_buffer,
        _base_t::ptr_base_controller_t SIXTRL_RESTRICT controller ) :
        st::ArchBase( arch_id, arch_str, config_str ),
        m_ptr_raw_arg_begin( nullptr ),
        m_ptr_cobj_cxx_buffer( nullptr ),
        m_ptr_cobj_c99_buffer( nullptr ),
        m_ptr_base_controller( controller ),
        m_arg_size( _base_t::size_type{ 0 } ),
        m_arg_capacity( _base_t::size_type{ 0 } ),
        m_needs_arg_buffer( needs_argument_buffer ),
        m_has_arg_buffer( false )
    {

    }

    /* --------------------------------------------------------------------- */

    bool ArgumentBase::doReserveArgumentBuffer(
        _base_t::size_type const required_arg_buffer_capacity )
    {
        return ( this->capacity() >= required_arg_buffer_capacity );
    }

    _base_t::status_t ArgumentBase::doUpdateRegions(
        _base_t::size_type const num_regions_to_update,
        _base_t::size_type const* SIXTRL_RESTRICT offsets,
        _base_t::size_type const* SIXTRL_RESTRICT lengths,
        void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values )
    {
        using size_t = _base_t::size_type;

        _base_t::status_t status = st::ARCH_STATUS_GENERAL_FAILURE;

        if( ( num_regions_to_update > _base_t::size_type{ 0 } ) &&
            ( offsets != nullptr ) && ( lengths != nullptr ) &&
            ( new_values != nullptr ) )
        {
            status = st::ARCH_STATUS_SUCCESS;
            size_t const arg_size = this->m_arg_size;

            size_t ii = size_t{ 0 };
            for( ; ii < num_regions_to_update ; ++ii )
            {
                size_t const offset = offsets[ ii ];
                size_t const length = lengths[ ii ];
                void const* SIXTRL_RESTRICT new_value = new_values[ ii ];

                if( ( new_value == nullptr ) || ( length == size_t{ 0 } ) ||
                    ( ( offset + length ) > arg_size ) )
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                    break;
                }

                unsigned char* dest = nullptr;

                if( this->usesCObjectsCxxBuffer() )
                {
                    dest = reinterpret_cast< unsigned char* >( static_cast<
                        uintptr_t >( this->ptrCObjectsCxxBuffer(
                            )->getDataBeginAddr() ) );
                }
                else if( this->usesCObjectsBuffer() )
                {
                    dest = reinterpret_cast< unsigned char* >( static_cast<
                    uintptr_t >( ::NS(Buffer_get_data_begin_addr)(
                        this->ptrCObjectsBuffer() ) ) );

                }
                else if( this->usesRawArgument() )
                {
                    dest = reinterpret_cast< unsigned char* >(
                        this->m_ptr_raw_arg_begin );
                }

                if( dest != nullptr )
                {
                    std::advance( dest, offset );
                    std::memcpy( dest, new_value, length );
                }
                else
                {
                    status = st::ARCH_STATUS_GENERAL_FAILURE;
                    break;
                }
            }
        }

        return status;
    }

    /* ----------------------------------------------------------------- */

    void ArgumentBase::doSetArgSize(
        _base_t::size_type const arg_size ) SIXTRL_NOEXCEPT
    {
        this->m_arg_size = arg_size;
    }

    void ArgumentBase::doSetArgCapacity(
        _base_t::size_type const arg_capacity ) SIXTRL_NOEXCEPT
    {
        this->m_arg_capacity = arg_capacity;
    }

    void ArgumentBase::doSetPtrControllerBase(
        _base_t::ptr_base_controller_t SIXTRL_RESTRICT ctrl
    ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_base_controller = ctrl;
    }

    void ArgumentBase::doSetBufferRef( _base_t::buffer_t const&
        SIXTRL_RESTRICT_REF buffer ) SIXTRL_NOEXCEPT
    {
        using c_buffer_t = _base_t::c_buffer_t;
        using buffer_t   = _base_t::buffer_t;

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
        const _base_t::c_buffer_t *const
        SIXTRL_RESTRICT ptr_c_buffer ) SIXTRL_NOEXCEPT
    {
        using c_buffer_t = _base_t::c_buffer_t;

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
