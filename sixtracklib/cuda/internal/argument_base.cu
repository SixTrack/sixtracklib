#include "sixtracklib/cuda/internal/argument_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <cuda.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"

#include "sixtracklib/cuda/internal/context_base.h"
#include "sixtracklib/cuda/wrappers/buffer_remap.h"

namespace SIXTRL_CXX_NAMESPACE
{
    CudaArgumentBase::~CudaArgumentBase() SIXTRL_NOEXCEPT
    {

    }

    bool CudaArgumentBase::write(
        CudaArgumentBase::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        bool success = false;

        if( 0 == this->doWriteAndRemapBuffer( buffer.getCApiPtr() ) )
        {
            this->doSetPtrCBuffer( buffer.getCApiPtr() );
            this->doSetPtrBuffer( &buffer );

            success = true;
        }

        return success;
    }

    bool CudaArgumentBase::write(
        CudaArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer )
    {
        bool success = false;

        if( 0 == this->doWriteAndRemapBuffer( ptr_c_buffer ) )
        {
            this->doSetPtrCBuffer( ptr_c_buffer );
            success = true;
        }

        return success;
    }

    bool CudaArgumentBase::write( void const* SIXTRL_RESTRICT arg_buffer_begin,
        CudaArgumentBase::size_type const arg_size )
    {
        bool success = false;

        using _this_t = CudaArgumentBase;
        using ptr_context_t = _this_t::ptr_context_t;
        using size_t = _this_t::size_type;

        ptr_context_t ptr_context = this->doGetPtrBaseContext();

        if( ( arg_size > size_t{ 0 } ) &&
            ( arg_buffer_begin != nullptr ) && ( ptr_context != nullptr ) &&
            ( ptr_context->hasSelectedNode() ) )
        {
            SIXTRL_ASSERT( this->size() <= this->capacity() );

            ::cudaError_t ret = cudaSuccess;

            if( this->cudaArgumentBuffer() != nullptr )
            {
                SIXTRL_ASSERT( this->size() > size_t{ 0 } );

                if( arg_size > this->capacity() )
                {
                    this->doResetArgBuffer( nullptr, size_t{ 0 } );
                }
                else
                {
                    success = true;
                }

                this->doSetArgSize( size_t{ 0 } );
            }

            if( this->cudaArgumentBuffer() == nullptr )
            {
                SIXTRL_ASSERT( this->size() == size_t{ 0 } );
                SIXTRL_ASSERT( this->capacity() == this->size() );
                unsigned char* arg_buffer = nullptr;

                ret = ::cudaMalloc( reinterpret_cast< void** >( &arg_buffer ),
                    arg_size );

                if( ret == cudaSuccess )
                {
                    this->doResetArgBuffer( arg_buffer, arg_size );
                    this->doSetArgSize( size_t{ 0 } );
                    success = true;
                }
            }

            if( success )
            {
                SIXTRL_ASSERT( ( this->cudaArgumentBuffer() != nullptr ) &&
                               ( this->capacity() >= arg_size ) &&
                               ( this->size() == size_t{ 0 } ) );

                this->doSetPtrBuffer( nullptr );
                this->doSetPtrCBuffer( nullptr );

                ret = ::cudaMemcpy( this->cudaArgumentBuffer(),
                    arg_buffer_begin, arg_size, cudaMemcpyHostToDevice );

                if( ret == cudaSuccess )
                {
                    this->doSetArgSize( arg_size );
                }
                else
                {
                    success = false;
                }
            }
        }

        return success;
    }

    bool CudaArgumentBase::read(
        CudaArgumentBase::buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        bool success = false;

        if( 0 == this->doReadAndRemapBuffer( buffer.getCApiPtr() ) )
        {
            success = true;
        }

        return success;
    }

    bool CudaArgumentBase::read(
        CudaArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer )
    {
        bool success = false;

        if( 0 == thsi->doReadAndRemapBuffer( ptr_c_buffer ) )
        {
            success = true;
        }

        return success;
    }

    bool CudaArgumentBase::read(
        void* SIXTRL_RESTRICT output_dest_begin,
        CudaArgumentBase::size_type const arg_size )
    {
        bool success = false;

        ptr_context_t ptr_context = this->doGetPtrBaseContext();

        if( ( arg_size > size_t{ 0 } ) && ( output_dest_begin != nullptr ) &&
            ( arg_size == this->size() ) && ( this->capacity() >= arg_size ) &&
            ( this->cudaArgumentBuffer() != nullptr ) &&
            ( ptr_context != nullptr ) && ( ptr_context->hasSelectedNode() ) )
        {
            cudaError_t ret = ::cudaMemcpy( output_dest_begin,
                this->cudaArgumentBuffer(), cudaMemcpyDeviceToHost );

            success = ( ret == cudaSuccess );
        }

        return success;
    }

    bool CudaArgumentBase::usesCObjectsBuffer()  const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_cobj_buffer == nullptr ) ||
            ( ( this->m_ptr_cobj_c_buffer != nullptr ) &&
              ( this->m_ptr_cobj_buffer->getCApiPtr() ==
                this->m_ptr_cobj_c_buffer ) ) );

        return ( this->m_ptr_cobj_buffer != nullptr );
    }

    CudaArgumentBase::buffer_t*
    CudaArgumentBase::ptrCObjectsBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cobj_buffer;
    }

    CudaArgumentBase::buffer_t& CudaArgumentBase::cobjectsBuffer() const
    {
        if( !this->usesCObjectsBuffer() )
        {
            throw std::runtime_error(
                "arg does not use a CObject buffer instance" );
        }

        return *this->ptrCObjectsBuffer();
    }

    bool CudaArgumentBase::usesCObjectsCBuffer() const SIXTRL_NOEXCEPT
    {
        SIXTRL_ASSERT( ( this->m_ptr_cobj_buffer == nullptr ) ||
            ( ( this->m_ptr_cobj_c_buffer != nullptr ) &&
              ( this->m_ptr_cobj_buffer->getCApiPtr() ==
                this->m_ptr_cobj_c_buffer ) ) );

        return ( this->m_ptr_cobj_c_buffer != nullptr );
    }

    CudaArgumentBase::c_buffer_t*
    CudaArgumentBase::ptrCObjectsCBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cobj_c_buffer;
    }

    bool CudaArgumentBase::usesRawArgumentBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->m_ptr_raw_arg_buffer != nullptr );
    }

    void* CudaArgumentBase::ptrRawArgumentBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_raw_arg_buffer;
    }

    CudaArgumentBase::ptr_cuda_arg_buffer_t
    CudaArgumentBase::cudaArgumentBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = CudaArgumentBase;
        using ptr_t   = _this_t::ptr_cuda_arg_buffer_t;

        return const_cast< ptr_t >( static_cast< _this_t const& >(
            *this ).cudaArgumentBuffer() );
    }

    CudaArgumentBase::ptr_const_cuda_arg_buffer_t
    CudaArgumentBase::cudaArgumentBuffer() const SIXTRL_NOEXCEPT
    {
        using size_t = CudaArgumentBase::size_type;

        SIXTRL_ASSERT( ( ( this->m_arg_buffer != nullptr ) &&
                         ( this->capacity() > size_t{ 0 } ) &&
                         ( this->capacity() >= this->size() ) ) ||
                       ( ( this->m_arg_buffer == nullptr ) &&
                         ( this->size() == this->capacity() ) &&
                         ( this->size() == size_t{ 0 } ) ) );

        return this->m_arg_buffer;
    }

    CudaArgumentBase::size_type
    CudaArgumentBase::size() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_size;
    }

    CudaArgumentBase::size_type
    CudaArgumentBase::capacity() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_capacity;
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        m_arg_buffer( nullptr ),
        m_ptr_raw_arg_buffer( nullptr ),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_c_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( CudaArgumentBase::size_type{ 0 } ),
        m_arg_capacity( CudaArgumentBase::size_type{ 0 } )
    {

    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::buffer_t& SIXTRL_RESTRICT_REF buffer,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        m_arg_buffer( nullptr ),
        m_ptr_raw_arg_buffer( nullptr ),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_c_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( CudaArgumentBase::size_type{ 0 } ),
        m_arg_capacity( CudaArgumentBase::size_type{ 0 } )
    {
        using size_t = CudaArgumentBase::size_type;
        using addr_t = CudaArgumentBase::buffer_t::address_t;

        size_t const buffer_size = buffer.size();

        if( ( buffer_size > size_t{ 0 } ) &&
            ( buffer.getDataStoreBeginAddr() != addr_t{ 0 } ) )
        {
            unsigned char* arg_buffer = nullptr;
            cudaError_t ret = ::cudaMalloc( reinterpret_cast< void** >(
                &arg_buffer, buffer_size ) );

            if( ret == cudaSuccess )
            {
                this->doResetArgBufferBaseImpl( arg_buffer, buffer_size );

                if( 0 == this->doWriteAndRemapBufferBaseImpl(
                    buffer.getCApiPtr() ) )
                {
                    this->doSetPtrCBuffer( buffer.getCApiPtr() );
                    this->doSetPtrBuffer( &buffer );
                }
            }
        }
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::c_buffer_t* SITRL_RESTRICT ptr_c_buffer,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        m_arg_buffer( nullptr ),
        m_ptr_raw_arg_buffer( nullptr ),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_c_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( CudaArgumentBase::size_type{ 0 } ),
        m_arg_capacity( CudaArgumentBase::size_type{ 0 } )
    {
        using size_t = CudaArgumentBase::size_type;
        using addr_t = CudaArgumentBase::buffer_t::address_t;

        size_t const buffer_size = ::NS(Buffer_get_size)( ptr_c_buffer );
        addr_t const addr = ::NS(Buffer_get_data_begin_addr)( ptr_c_buffer );

        if( ( ptr_c_buffer != nullptr ) && ( buffer_size > size_t{ 0 } ) &&
            ( addr != addr_t{ 0 } ) )
        {
            unsigned char* arg_buffer = nullptr;
            cudaError_t ret = ::cudaMalloc( reinterpret_cast< void** >(
                &arg_buffer, buffer_size ) );

            if( ret == cudaSuccess )
            {
                this->doResetArgBufferBaseImpl( arg_buffer, buffer_size );

                if( 0 == this->doWriteAndRemapBufferBaseImpl( ptr_c_buffer ) )
                {
                    this->doSetPtrCBuffer( ptr_c_buffer );
                }
            }
        }
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::size_type const arg_size,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        m_arg_buffer( nullptr ),
        m_ptr_raw_arg_buffer( nullptr ),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_c_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( CudaArgumentBase::size_type{ 0 } ),
        m_arg_capacity( CudaArgumentBase::size_type{ 0 } )
    {
        using size_t = CudaArgumentBase::size_type;

        if( arg_size > size_t{ 0 } )
        {
            unsigned char* arg_buffer = nullptr;
            cudaError_t ret = ::cudaMalloc( reinterpret_cast< void** >(
                &arg_buffer, arg_size ) );

            if( ret == cudaSuccess )
            {
                this->doResetArgBufferBaseImpl( arg_buffer, arg_size );
            }
        }
    }

    CudaArgumentBase::CudaArgumentBase(
        void const* SIXTRL_RESTRICT raw_argument_begin,
        CudaArgumentBase::size_type const arg_size,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context )  :
        m_arg_buffer( nullptr ),
        m_ptr_raw_arg_buffer( nullptr ),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_c_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( CudaArgumentBase::size_type{ 0 } ),
        m_arg_capacity( CudaArgumentBase::size_type{ 0 } )
    {
        using size_t = CudaArgumentBase::size_type;

        if( ( arg_size > size_t{ 0 } ) && ( raw_argument_begin != nullptr ) )
        {
            unsigned char* arg_buffer = nullptr;

            cudaError_t ret = ::cudaMalloc( reinterpret_cast< void** >(
                &arg_buffer ), arg_size );

            if( ret == cudaSuccess )
            {
                this->doResetArgBufferBaseImpl( arg_buffer, arg_size );

                ret = ::cudaMempcy( this->cudaARgumentBuffer(),
                    raw_argument_begin, arg_size, cudaMemcpyHostToDevice );

                if( ret == cudaSuccess )
                {
                    this->doSetRawArgumentBuffer( raw_argument_begin );
                    this->doSetArgSize( arg_size );
                }
                else
                {
                    this->doResetArgBufferBaseImpl( nullptr, size_t{ 0 } );
                }
            }
        }
    }

    int CudaArgumentBase::doWriteAndRemapBuffer(
        CudaArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer )
    {
        return this->doWriteAndRemapBufferBaseImpl( ptr_buffer );
    }

    int CudaArgumentBase::doReadAndRemapBuffer(
        CudaArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer )
    {
        return this->doWriteAndRemapBufferBaseImpl( ptr_buffer );
    }

    void CudaArgumentBase::doResetArgBuffer(
        unsigned char* SIXTRL_RESTRICT new_arg_buffer,
        CudaArgumentBase::size_type const buffer_capacity )
    {
        this->doResetArgBufferBaseImpl( new_arg_buffer, buffer_capacity );
    }

    void CudaArgumentBase::doSetPtrContext( CudaArgumentBase::ptr_context_t
        SIXTRL_RESTRICT ptr_context ) SIXTRL_NOEXCEPT
    {

    }

    void CudaArgumentBase::doSetPtrBuffer( CudaArgumentBase::buffer_t*
        SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT
    {

    }

    void CudaArgumentBase::doSetPtrCBuffer( CudaArgumentBase::c_buffer_t*
        SIXTRL_RESTRICT ptr_c_buffer ) SIXTRL_NOEXCEPT
    {

    }

    void CudaArgumentBase::doSetArgSize(
        CudaArgumentBase::size_type const arg_size ) SIXTRL_NOEXCEPT
    {

    }

    void CudaArgumentBase::doSetRawArgumentBuffer(
        void* SIXTRL_RESTRICT raw_arg_begin ) SIXTRL_NOEXCEPT
    {
        this->m_ptr_raw_arg_buffer = raw_arg_begin;
    }

    CudaArgumentBase::ptr_const_context_t
    CudaArgumentBase::doGetPtrBaseContext() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_context;
    }

    CudaArgumentBase::ptr_context_t
    CudaArgumentBase::doGetPtrBaseContext() SIXTRL_NOEXCEPT
    {
        return this->m_ptr_context;
    }

    int CudaArgumentBase::doReadAndRemapBufferBaseImpl(
        CudaArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer )
    {
        int success = -1;

        using size_t = CudaArgumentBase::size_type;

        size_t const buffer_capacity = ::NS(Buffer_get_capacity)( ptr_buffer );

        if( ( ptr_buffer != nullptr ) &&
            ( ::NS(Buffer_get_data_begin)( ptr_buffer ) != nullptr ) &&
            ( this->cudaArgumentBuffer()  != nullptr ) &&
            ( this->doGetPtrBaseContext() != nullptr ) &&
            ( this->doGetPtrBaseContext()->hasSelectedNode() ) &&
            ( this->usesCObjectsCBuffer() ) &&
            ( this->size() > size_t{ 0 } ) &&
            ( this->size() <= capacity ) )
        {
            cudaError_t ret = ::cudaMemcpy(
                ::NS(Buffer_get_data_begin)( ptr_buffer ),
                this->cudaArgumentBuffer(), this->size(),
                cudaMemcpyDeviceToHost );

            if( ret == cudaSuccess )
            {
                success = ::NS(Buffer_remap)( ptr_buffer );
            }
        }

        return success;
    }

    int CudaArgumentBase::doWriteAndRemapBufferBaseImpl(
        CudaArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_buffer )
    {
        int success = -1;

        using size_t = CudaArgumentBase::size_type;

        size_t const buffer_size = ::NS(Buffer_get_size)( ptr_buffer );
        size_t const slot_size   = ::NS(Buffer_get_slot_size)( ptr_buffer );

        unsigned char const* buffer_data_begin =
            ::NS(Buffer_get_data_begin)( ptr_buffer );

        if( ( ptr_buffer != nullptr ) && ( buffer_size > size_t{ 0 } ) &&
            ( slot_size > size_t{ 0 } ) && ( buffer_data_begin != nullptr ) &&
            ( !::NS(ManagedBuffer_needs_remapping)(
                    buffer_data_begin, slot_size ) ) &&
            ( this->doGetPtrBaseContext() != nullptr ) &&
            ( this->doGetPtrBaseContext()->hasSelectedNode() ) )
        {
            cudaError_t ret = cudaSuccess;

            if( this->cudaArgumentBuffer() != nullptr )
            {
                SIXTRL_ASSERT( this->size() <= this->capacity() );
                SIXTRL_ASSERT( thsi->capacity() > size_t{ 0 } );

                if( this->capacity() < buffer_size )
                {
                    unsigned char* arg_buffer = nullptr;

                }
            }

            ( this->size() > size_t{ 0 } ) &&
            ( this->size() <= capacity ) )
        {
            cudaError_t ret = ::cudaMemcpy(
                ::NS(Buffer_get_data_begin)( ptr_buffer ),
                this->cudaArgumentBuffer(), this->size(),
                cudaMemcpyDeviceToHost );

            if( ret == cudaSuccess )
            {
                success = ::NS(Buffer_remap)( ptr_buffer );
            }
        }

        return success;
    }

    void CudaArgumentBase::doResetArgBufferBaseImpl(
        unsigned char* new_arg_buffer ) SIXTRL_NOEXCEPT
    {

    }
}

/* end: sixtracklib/cuda/internal/argument_base.cu */
