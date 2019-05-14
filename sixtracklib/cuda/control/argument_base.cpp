#include "sixtracklib/cuda/control/argument_base.hpp"

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
#include "sixtracklib/common/control/debug_register.h"
#include "sixtracklib/common/control/argument_base.hpp"
#include "sixtracklib/common/output/elem_by_elem_config.h"
#include "sixtracklib/cuda/controller.hpp"

namespace st = SIXTRL_CXX_NAMESPACE;

namespace SIXTRL_CXX_NAMESPACE
{
    CudaArgumentBase::~CudaArgumentBase() SIXTRL_NOEXCEPT
    {
        this->doDeleteCudaArgumentBuffer();
    }

    bool CudaArgumentBase::hasCudaArgBuffer() const SIXTRL_NOEXCEPT
    {
        return ( ( this->hasArgumentBuffer() ) &&
                 ( this->m_arg_buffer != nullptr ) );
    }

    CudaArgumentBase::cuda_arg_buffer_t
    CudaArgumentBase::cudaArgBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_arg_buffer;
    }

    CudaArgumentBase::cuda_const_arg_buffer_t
    CudaArgumentBase::cudaArgBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_buffer;
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::size_type const arg_buffer_capacity,
        CudaController* SIXTRL_RESTRICT ctrl ) :
        ArgumentBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, nullptr, true, ctrl ),
        m_arg_buffer( nullptr )
    {
        this->doReserveArgumentBufferCudaBaseImpl( arg_buffer_capacity );
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaController* SIXTRL_RESTRICT ctrl ) :
        ArgumentBase( st::ARCHITECTURE_CUDA,
            SIXTRL_ARCHITECTURE_CUDA_STR, nullptr, true, ctrl ),
        m_arg_buffer( nullptr )
    {

    }

    void CudaArgumentBase::doDeleteCudaArgumentBuffer() SIXTRL_NOEXCEPT
    {
        using _this_t = st::CudaArgumentBase;
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

    void CudaArgumentBase::doResetCudaArgumentBuffer(
        CudaArgumentBase::cuda_arg_buffer_t SIXTRL_RESTRICT arg_buffer,
        CudaArgumentBase::size_type const capacity )
    {
        using size_t = CudaArgumentBase::size_type;

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

    /* ----------------------------------------------------------------- */

    bool CudaArgumentBase::doReserveArgumentBuffer(
        CudaArgumentBase::size_type const required_buffer_size )
    {
        return this->doReserveArgumentBufferCudaBaseImpl(
            required_buffer_size );
    }

    bool CudaArgumentBase::doReserveArgumentBufferCudaBaseImpl(
        CudaArgumentBase::size_type const required_buffer_size )
    {
        bool success = false;

        using _this_t      = st::CudaArgumentBase;
        using size_t       = _this_t::size_type;
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

    /* ----------------------------------------------------------------- */

    CudaArgumentBase::cuda_arg_buffer_t CudaArgumentBase::CudaAllocArgBuffer(
        CudaArgumentBase::size_type const capacity )
    {
        CudaArgumentBase::cuda_arg_buffer_t arg_buffer = nullptr;

        if( capacity > CudaArgumentBase::size_type{ 0 } )
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

    void CudaArgumentBase::CudaFreeArgBuffer(
        CudaArgumentBase::cuda_arg_buffer_t SIXTRL_RESTRICT arg_buffer )
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
    CudaArgumentBase::cudaArgBufferAsCObjectsDataBegin() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BUFFER_DATAPTR_DEC unsigned char* >(
            static_cast< CudaArgumentBase const& >( *this
                ).cudaArgBufferAsCObjectsDataBegin() );
    }

    SIXTRL_HOST_FN SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*
    CudaArgumentBase::cudaArgBufferAsCObjectsDataBegin() const SIXTRL_NOEXCEPT
    {
        using ptr_t = SIXTRL_BUFFER_DATAPTR_DEC unsigned char const*;

        if( ( this->hasArgumentBuffer() ) &&
            ( ( this->usesCObjectsBuffer() ) ||
              ( this->usesCObjectsCxxBuffer() ) ) &&
            ( this->size() > CudaArgumentBase::size_type{ 0 } ) )
        {
            return reinterpret_cast< ptr_t >( this->cudaArgBuffer() );
        }

        return nullptr;
    }

    SIXTRL_BUFFER_DATAPTR_DEC CudaArgumentBase::debug_register_t*
    CudaArgumentBase::cudaArgBufferAsPtrDebugRegister() SIXTRL_NOEXCEPT
    {
        return const_cast< SIXTRL_BUFFER_DATAPTR_DEC
            CudaArgumentBase::debug_register_t* >( static_cast<
                CudaArgumentBase const& >( *this
                    ).cudaArgBufferAsPtrDebugRegister() );
    }

    SIXTRL_BUFFER_DATAPTR_DEC CudaArgumentBase::debug_register_t const*
    CudaArgumentBase::cudaArgBufferAsPtrDebugRegister() const SIXTRL_NOEXCEPT
    {
        using dbg_register_t = CudaArgumentBase::debug_register_t;
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
    CudaArgumentBase::elem_by_elem_config_t*
    CudaArgumentBase::cudaArgBufferAsElemByElemByElemConfig() SIXTRL_NOEXCEPT
    {
        return const_cast< CudaArgumentBase::elem_by_elem_config_t* >(
            static_cast< CudaArgumentBase const& >( *this
                ).cudaArgBufferAsElemByElemByElemConfig() );
    }

    SIXTRL_ELEM_BY_ELEM_CONFIG_ARGPTR_DEC
    CudaArgumentBase::elem_by_elem_config_t const*
    CudaArgumentBase::cudaArgBufferAsElemByElemByElemConfig() const SIXTRL_NOEXCEPT
    {
        using elem_by_elem_conf_t = CudaArgumentBase::elem_by_elem_config_t;
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

#endif /* c++, host */

/* end: sixtracklib/cuda/control/argument_base.cpp */
