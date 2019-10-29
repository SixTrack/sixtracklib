#include "sixtracklib/cuda/argument.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <cuda_runtime_api.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.hpp"
#include "sixtracklib/common/control/controller_base.hpp"

#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/controller.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    CudaArgument::CudaArgument( CudaController* SIXTRL_RESTRICT ctrl ) :
        ArgumentBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, nullptr, true, ctrl ),
        m_arg_buffer( nullptr )
    {

    }

    CudaArgument::~CudaArgument()
    {
        st::CudaArgument::CudaFreeArgBuffer( this->m_arg_buffer );
        this->m_arg_buffer = nullptr;
    }

    CudaArgument::CudaArgument(
        CudaArgument::buffer_t const& SIXTRL_RESTRICT_REF buffer,
        CudaController* SIXTRL_RESTRICT ctrl ) :
        ArgumentBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, nullptr, true, ctrl ),
        m_arg_buffer( nullptr )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        size_t const buffer_size = buffer.size();
        this->doReserveArgumentBufferCudaBaseImpl( buffer_size );

        void const* buffer_data_begin = buffer.dataBegin< void const* >();

        if( ( buffer_size > size_t{ 0 } ) &&
            ( this->capacity() >= buffer_size ) &&
            ( this->cudaController() == ctrl ) &&
            ( ctrl != nullptr ) && ( ctrl->readyForSend() ) &&
            ( ctrl->readyForRemap() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );

            status_t status = this->cudaController()->sendMemory(
                this->cudaArgBuffer(), buffer_data_begin, buffer_size );

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                status = ctrl->remap(
                    this->cudaArgBuffer(), buffer.getSlotSize() );
            }

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                this->doSetArgSize( buffer_size );
                this->doSetBufferRef( buffer );
            }
        }
    }

    CudaArgument::CudaArgument(
        const CudaArgument::c_buffer_t *const SIXTRL_RESTRICT ptr_buffer,
        CudaController* SIXTRL_RESTRICT ctrl ) :
        ArgumentBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, nullptr, true, ctrl ),
        m_arg_buffer( nullptr )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        size_t const buffer_size = ::NS(Buffer_get_size)( ptr_buffer );
        this->doReserveArgumentBufferCudaBaseImpl( buffer_size );

        void const*  buffer_data_begin =
            ::NS(Buffer_get_const_data_begin)( ptr_buffer );

        if( ( buffer_size > size_t{ 0 } ) &&
            ( this->capacity() >= buffer_size ) &&
            ( this->cudaController() == ctrl ) &&
            ( ctrl != nullptr ) && ( ctrl->readyForSend() ) &&
            ( ctrl->readyForRemap() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );

            status_t status = this->cudaController()->sendMemory(
                this->cudaArgBuffer(), buffer_data_begin, buffer_size );

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                status = ctrl->remap( this->cudaArgBuffer(),
                    ::NS(Buffer_get_slot_size( ptr_buffer ) ) );
            }

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                this->doSetArgSize( buffer_size );
                this->doSetPtrCBuffer( ptr_buffer );
            }
        }
    }

    CudaArgument::CudaArgument( CudaArgument::size_type const capacity,
        CudaController* SIXTRL_RESTRICT ctrl )  :
        ArgumentBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, nullptr, true, ctrl ),
        m_arg_buffer( nullptr )
    {
        this->doReserveArgumentBufferCudaBaseImpl( capacity );
    }

   CudaArgument::CudaArgument(
        const void *const SIXTRL_RESTRICT raw_arg_begin,
        CudaArgument::size_type const raw_arg_length,
        CudaController* SIXTRL_RESTRICT ctrl )  :
        ArgumentBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, nullptr, true, ctrl ),
        m_arg_buffer( nullptr )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        this->doReserveArgumentBufferCudaBaseImpl( raw_arg_length );

        if( ( raw_arg_length > size_t{ 0 } ) &&
            ( this->cudaController() != nullptr ) &&
            ( this->cudaController() == ctrl ) && ( ctrl->readyForSend() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );
            SIXTRL_ASSERT( this->capacity() >= raw_arg_length );

            status_t status = this->cudaController()->sendMemory(
                this->cudaArgBuffer(), raw_arg_begin, raw_arg_length );

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                this->doSetArgSize( raw_arg_length );
                this->doSetPtrRawArgument( raw_arg_begin );
            }
        }
    }

    /* --------------------------------------------------------------------- */

    CudaArgument::ptr_cuda_controller_t
    CudaArgument::cudaController() SIXTRL_NOEXCEPT
    {
        using _this_t   = CudaArgument;
        using ptr_ctrl_t = _this_t::ptr_cuda_controller_t;

        return const_cast< ptr_ctrl_t >( static_cast< _this_t const& >(
            *this ).cudaController() );
    }

    CudaArgument::ptr_const_cuda_controller_t
    CudaArgument::cudaController() const SIXTRL_NOEXCEPT
    {
        using cuda_ctrl_t = st::CudaController;

        return ( this->ptrControllerBase() != nullptr )
            ? this->ptrControllerBase()->asDerivedController< cuda_ctrl_t >(
                    st::ARCHITECTURE_CUDA ) : nullptr;
    }

    /* --------------------------------------------------------------------- */

    bool CudaArgument::hasCudaArgBuffer() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasArgumentBuffer() ) &&
                 ( this->m_arg_buffer != nullptr ) );
    }

    CudaArgument::cuda_arg_buffer_t
    CudaArgument::cudaArgBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_arg_buffer;
    }

    CudaArgument::cuda_const_arg_buffer_t
    CudaArgument::cudaArgBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_buffer;
    }

    void CudaArgument::doDeleteCudaArgumentBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = st::CudaArgument;
        using size_t = _this_t::size_type;

        _this_t::CudaFreeArgBuffer( this->m_arg_buffer );

        this->m_arg_buffer = nullptr;
        this->doSetArgCapacity( size_t{ 0 } );
        this->doSetArgSize( size_t{ 0 } );
        this->doResetPtrCxxBuffer();
        this->doSetPtrCBuffer( nullptr );
        this->doSetPtrRawArgument( nullptr );

        return;
    }

    void CudaArgument::doResetCudaArgumentBuffer(
        CudaArgument::cuda_arg_buffer_t SIXTRL_RESTRICT arg_buffer,
        CudaArgument::size_type const capacity )
    {
        using size_t = CudaArgument::size_type;

        if( this->m_arg_buffer != arg_buffer )
        {
            if( this->m_arg_buffer != nullptr )
            {
                this->doDeleteCudaArgumentBuffer();
            }

            if( ( ( arg_buffer != nullptr ) && ( capacity >  size_t{ 0 } ) ) ||
                ( ( arg_buffer == nullptr ) && ( capacity == size_t{ 0 } ) ) )
            {
                this->m_arg_buffer = arg_buffer;
                this->doSetArgCapacity( capacity );
            }
        }

        return;
    }

    /* --------------------------------------------------------------------- */

    bool CudaArgument::doReserveArgumentBuffer(
        CudaArgument::size_type const required_buffer_size )
    {
        return this->doReserveArgumentBufferCudaBaseImpl(
            required_buffer_size );
    }

    bool CudaArgument::doReserveArgumentBufferCudaBaseImpl(
        CudaArgument::size_type const required_buffer_size )
    {
        bool success = false;

        using _this_t = st::CudaArgument;
        using size_t = _this_t::size_type;
        using arg_buffer_t = _this_t::cuda_arg_buffer_t;

        if( this->capacity() < required_buffer_size )
        {
            arg_buffer_t arg_buffer =
                _this_t::CudaAllocArgBuffer( required_buffer_size );

            if( arg_buffer != nullptr )
            {
                this->doResetCudaArgumentBuffer(
                    arg_buffer, required_buffer_size );

                success = true;
            }
        }
        else
        {
            success = true;
        }

        if( ( success ) && ( !this->hasArgumentBuffer() ) )
        {
            this->doSetHasArgumentBufferFlag( true );
        }

        return success;
    }

    /* --------------------------------------------------------------------- */

    CudaArgument::cuda_arg_buffer_t CudaArgument::CudaAllocArgBuffer(
        CudaArgument::size_type const capacity )
    {
        CudaArgument::cuda_arg_buffer_t arg_buffer = nullptr;

        if( capacity > CudaArgument::size_type{ 0 } )
        {
            ::cudaError_t const ret =
                ::cudaMalloc( ( void** )&arg_buffer, capacity );

            if( ret != ::cudaSuccess )
            {
                if( arg_buffer != nullptr )
                {
                    ::cudaFree( arg_buffer );
                    arg_buffer = nullptr;
                }
            }
        }

        return arg_buffer;
    }

    void CudaArgument::CudaFreeArgBuffer(
        CudaArgument::cuda_arg_buffer_t SIXTRL_RESTRICT arg_buffer )
    {
        if( arg_buffer != nullptr )
        {
            ::cudaError_t const err = ::cudaFree( arg_buffer );
            SIXTRL_ASSERT( err == ::cudaSuccess );
            ( void )err;

            arg_buffer = nullptr;
        }
    }

    SIXTRL_BUFFER_DATAPTR_DEC unsigned char*
    CudaArgument::cudaArgBufferAsCObjectsDataBegin() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BUFFER_DATAPTR_DEC unsigned char* >(
            static_cast< CudaArgument const& >( *this
                ).cudaArgBufferAsCObjectsDataBegin() );
    }

    SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
    CudaArgument::cudaArgBufferAsCObjectsDataBegin() const SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*;

        if( ( this->hasArgumentBuffer() ) &&
            ( ( this->usesCObjectsBuffer() ) ||
              ( this->usesCObjectsCxxBuffer() ) ) &&
            ( this->size() > CudaArgument::size_type{ 0 } ) )
        {
            return reinterpret_cast< ptr_t >( this->cudaArgBuffer() );
        }

        return nullptr;
    }

    SIXTRL_BUFFER_DATAPTR_DEC CudaArgument::debug_register_t*
    CudaArgument::cudaArgBufferAsPtrDebugRegister() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BUFFER_DATAPTR_DEC
            CudaArgument::debug_register_t* >( static_cast<
                CudaArgument const& >( *this
                    ).cudaArgBufferAsPtrDebugRegister() );
    }

    SIXTRL_BUFFER_DATAPTR_DEC CudaArgument::debug_register_t const*
    CudaArgument::cudaArgBufferAsPtrDebugRegister() const SIXTRL_NOEXCEPT
    {
        using dbg_register_t = CudaArgument::debug_register_t;
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC dbg_register_t const*;

        if( ( this->hasArgumentBuffer() ) &&
            ( this->usesRawArgument() ) &&
            ( this->size() == sizeof( dbg_register_t ) ) )
        {
            return reinterpret_cast< ptr_t >( this->cudaArgBuffer() );
        }

        return nullptr;
    }

    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
    CudaArgument::elem_by_elem_config_t*
    CudaArgument::cudaArgBufferAsElemByElemByElemConfig() SIXTRL_NOEXCEPT
    {
        return const_cast< CudaArgument::elem_by_elem_config_t* >(
            static_cast< CudaArgument const& >( *this
                ).cudaArgBufferAsElemByElemByElemConfig() );
    }

    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
    CudaArgument::elem_by_elem_config_t const*
    CudaArgument::cudaArgBufferAsElemByElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        using elem_by_elem_conf_t = CudaArgument::elem_by_elem_config_t;
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC elem_by_elem_conf_t const*;

        if( ( this->hasArgumentBuffer() ) &&
            ( this->usesRawArgument() ) &&
            ( this->size() == sizeof( elem_by_elem_config_t ) ) )
        {
            return reinterpret_cast< ptr_t >( this->cudaArgBuffer() );
        }

        return nullptr;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/argument.cpp */
