#include "sixtracklib/common/context/argument_base.cpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/buffer.hpp"

#include "sixtracklib/common/context/context_base.h"

namespace SIXTRL_CXX_NAMESPACE
{
    bool ArgumentBase::write()
    {
        bool success = false;

        if( this->usesCObjectsCxxBuffer() )
        {
            success = this->write( this->cobjectsCxxBuffer() );
        }
        else if( this->usesCObjectsBuffer() )
        {
            success = this->write( this->ptrCObjectsBuffer() );
        }
        else if( this->usesRawArgument() )
        {
            success = this->write( this->ptrRawArgument(), this->size() );
        }

        return success;
    }

    bool ArgumentBase::write(
        ArgumentBase::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        bool success = false;

        if( this->doWriteAndRemapCObjectBuffer( buffer.getCApiPtr() ) )
        {
            if( !this->usesCObjectsBuffer() )
            {
                this->doSetPtrC99Buffer( buffer.getCApiPtr() );

                if( !this->usesCObjectsCxxBuffer() )
                {
                    this->doSetPtrCxxBuffer( &buffer );
                }
            }

            success = true;
        }

        return success;
    }

    bool ArgumentBase::write(
        ArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer )
    {
        bool success = false;

        if( this->doWriteAndRemapCObjectBuffer( ptr_cbuffer ) )
        {
            if( !this->usesCObjectsBuffer() )
            {
                this->doSetPtrC99Buffer( ptr_cbuffer );
            }

            success = true;
        }

        return success;
    }

    bool ArgumentBase::write( void const* SIXTRL_RESTRICT arg_begin,
        ArgumentBase::size_type const arg_size )
    {
        using size_t = ArgumentBase::size_type;

        bool success = false;

        if( ( arg_begin != nullptr ) && ( arg_size > size_t{ 0 } ) )
        {
            if( this->usesCObjectsCxxBuffer() )
            {
                this->doSetPtrCxxBuffer( nullptr );
            }

            if( this->usesCObjectsBuffer() )
            {
                this->doSetPtrC99Buffer( nullptr );
            }

            success = true;

            if( this->requiresArgumentBuffer() )
            {
                if( ( !this->hasArgumentBuffer() ) ||
                    ( this->capacity() < arg_size ) )
                {
                    success = ( 0 == this->doReserveArgumentBuffer(
                        arg_size ) );
                }

                success &= ( ( this->hasArgumentBuffer() ) &&
                             ( this->capacity() >= arg_size ) );
            }

            if( success )
            {
                success = ( 0 == this->doTransferBufferToDevice(
                    arg_begin, arg_size ) );
            }

            if( success )
            {
                this->doSetPtrRawArgument( arg_begin );
            }
        }

        return success;
    }

    bool ArgumentBase::read()
    {
        bool success = false;

        if( this->usesCObjectsCxxBuffer() )
        {
            success = this->read( this->cobjectsCxxBuffer() );
        }
        else if( this->usesCObjectsBuffer() )
        {
            success = this->read( this->ptrCObjectsBuffer() );
        }
        else if( this->usesRawArgument() )
        {
            success = this->read( this->ptrRawArgument(), this->size() );
        }

        return success;
    }

    bool ArgumentBase::read(
        ArgumentBase::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        bool succcess = false;

        if( this->doReadAndRemapCObjectBuffer( buffer.getCApiPtr() ) )
        {
            if( !this->usesCObjectsBuffer() )
            {
                this->doSetPtrC99Buffer( buffer.getCApiPtr() );

                if( !this->usesCObjectsCxxBuffer() )
                {
                    this->doSetPtrCxxBuffer( &buffer );
                }
            }

            success = true;
        }

        return success;
    }

    bool ArgumentBase::read( ArgumentBase::c_buffer_t*
        SIXTRL_RESTRICT ptr_c_buffer )
    {
        bool succcess = false;

        if( this->doReadAndRemapCObjectBuffer( ptr_c_buffer ) )
        {
            if( !this->usesCObjectsBuffer() )
            {
                this->doSetPtrC99Buffer( buffer.getCApiPtr() );
            }

            success = true;
        }

        return success;
    }

    bool ArgumentBase::read( void* SIXTRL_RESTRICT arg_begin,
        ArgumentBase::size_type const arg_size )
    {
        bool success = false;

        using size_t = ArgumentBase::size_type;

        if( ( arg_begin != nullptr ) && ( arg_size >= this->size() ) &&
            ( !this->usesCObjectsBuffer() ) )
        {
            success = ( 0 == this->doTransferBufferFromDevice(
                arg_begin, this->size() ) );

            if( ( success ) && ( !this->usesRawArgument() ) )
            {
                this->doSetPtrRawArgument( arg_begin );
            }
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

    ArgumentBase::ptr_context_t ArgumentBase::ptrBaseContext() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_context;
    }

    ArgumentBase::ptr_const_context_t
    ArgumentBase::ptrBaseContext() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_base_context;
    }

    ArgumentBase::ArgumentBase(
        bool const needs_argument_buffer,
        ArgumentBase::ptr_context_t SIXTRL_RESTRICT context ) SIXTRL_NOEXCEPT :
        m_ptr_raw_arg_begin( nullptr ),
        m_ptr_cobj_cxx_buffer( nullptr ),
        m_ptr_cobj_c99_buffer( nullptr ),
        m_ptr_base_context( context ),
        m_arg_size( ArgumentBase::size_type{ 0 } ),
        m_arg_capacity( ArgumentBase::size_type{ 0 } ),
        m_needs_arg_buffer( needs_argument_buffer ),
        m_has_arg_buffer( false )
    {

    }

    /* --------------------------------------------------------------------- */

    int ArgumentBase::doReserveArgumentBuffer( ArgumentBase::size_type const )
    {
        return int{ -1 };
    }

    int ArgumentBase::doTransferBufferToDevice(
        void const* SIXTRL_RESTRICT, ArgumentBase::size_type const )
    {
        return int{ -1 };
    }

    int ArgumentBase::doTransferBufferFromDevice(
        void* SIXTRL_RESTRICT, ArgumentBase::size_type const )
    {
        return int{ -1 };
    }

    int ArgumentBase::doRemapCObjectBufferAtDevice()
    {
        return int{ -1 };
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

    void ArgumentBase::doSetPtrContext( ArgumentBase::ptr_context_t
        SIXTRL_RESTRICT ptr_context ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_base_context = ptr_context;
    }

    void ArgumentBase::doSetPtrCxxBuffer( ArgumentBase::buffer_t*
        SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_cobj_cxx_buffer = ptr_buffer;
    }

    void ArgumentBase::doSetPtrC99Buffer( ArgumentBase::c_buffer_t*
        SIXTRL_RESTRICT ptr_c_buffer ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_cobj_c99_buffer = ptr_c_buffer;
    }

    void ArgumentBase::doSetPtrRawArgument(
        void* SIXTRL_RESTRICT raw_arg_begin ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_raw_arg_begin = raw_arg_begin;
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

    /* ===================================================================== */

    bool ArgumentBase::doWriteAndRemapCObjectBuffer(
            ArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer )
    {
        using size_t = ArgumentBase::size_type;
        using source_ptr_t = unsigned char const*;

        bool success = false;

        size_t const buf_size = buffer.size();
        source_ptr_t source_begin = buffer.dataBegin< source_ptr_t >();

        if( ( source_begin != nullptr ) && ( buf_size > size_t{ 0 } ) )
        {
            if( this->usesRawArgument() )
            {
                this->doSetPtrRawArgument( nullptr );
            }

            success = true;

            if( this->requiresArgumentBuffer() )
            {
                if( ( !this->hasArgumentBuffer() ) ||
                    ( this->capacity() < buf_size ) )
                {
                    success = (
                        0 == this->doReserveArgumentBuffer( buf_size ) );
                }

                success &= ( ( this->hasArgumentBuffer() ) &&
                             ( this->capacity() >= buf_size ) );
            }

            if( success )
            {
                success = ( 0 == this->doTransferBufferToDevice(
                    source_begin, buf_size ) );
            }

            if( success )
            {
                success = ( 0 == this->doRemapCObjectBufferAtDevice() );
            }
        }

        return success;
    }

    bool ArgumentBase::doReadAndRemapCObjectBuffer(
            ArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer )
    {
        using size_t = ArgumentBase::size_type;
        using dest_ptr_t = unsigned char*;

        bool success = false;

        size_t const buf_capacity = ::NS(Buffer_get_capacity)( ptr_buffer );
        dest_ptr_t dest_begin = ::NS(Buffer_get_data_begin)( ptr_buffer );

        if( ( dest_begin != nullptr ) && ( buf_capcity >= this->size() ) )
        {
            if( this->usesRawArgument() )
            {
                this->doSetPtrRawArgument( nullptr );
            }

            success = ( 0 == this->doTransferBufferFromDevice(
                dest_begin, this->size() ) );

            if( ( success ) &&
                ( ::NS(ManagedBuffer_needs_remapping)(
                    dest_begin, ::NS(Buffer_get_slot_size)( ptr_buffer ) ) ) )
            {
                success = ( 0 == ::NS(Buffer_remap)( ptr_buffer ) );
            }
        }

        return success;
    }
}

/* end: sixtracklib/common/context/argument_base.cpp */
