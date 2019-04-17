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
#include "sixtracklib/common/context/argument_base.h"
#include "sixtracklib/common/context/context_base.h"

#include "sixtracklib/cuda/internal/context_base.h"
#include "sixtracklib/cuda/wrappers/argument_operations.h"

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
        ContextOnNodesBase* SIXTRL_RESTRICT ctx ) :
        ArgumentBase( SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_CUDA,
            SIXTRL_CONTEXT_TYPE_CUDA_STR, true, ctx ),
        m_arg_buffer( nullptr )
    {
        this->doReserveArgumentBufferCudaBaseImpl( arg_buffer_capacity );
    }

    CudaArgumentBase::CudaArgumentBase(
        ContextOnNodesBase* SIXTRL_RESTRICT ctx ) :
        ArgumentBase( SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_CUDA,
            SIXTRL_CONTEXT_TYPE_CUDA_STR, true, ctx ),
        m_arg_buffer( nullptr )
    {

    }

    void CudaArgumentBase::doDeleteCudaArgumentBuffer() SIXTRL_NOEXCEPT
    {
        using size_t = CudaArgumentBase::size_type;

        ::NS(CudaArgument_free_arg_buffer)( this->m_arg_buffer );

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

        using size_t = CudaArgumentBase::size_type;
        using arg_buffer_t = CudaArgumentBase::cuda_arg_buffer_t;

        if( this->capacity() < required_buffer_size )
        {
            arg_buffer_t arg_buffer = ::NS(CudaArgument_alloc_arg_buffer)(
                required_buffer_size );

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
}

bool NS(CudaArgument_has_cuda_arg_buffer)(
    const ::NS(CudaArgumentBase) *const SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->hasCudaArgBuffer() : false;
}

::NS(cuda_arg_buffer_t) NS(CudaArgument_get_cuda_arg_buffer)(
    ::NS(CudaArgumentBase)* SIXTRL_RESTRICT arg )
{
    return ( arg != nullptr ) ? arg->cudaArgBuffer() : nullptr;
}

/* end: sixtracklib/cuda/internal/argument_base.cu */
