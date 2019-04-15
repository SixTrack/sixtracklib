#include "sixtracklib/cuda/argument.h"

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
#include "sixtracklib/cuda/internal/argument_base.h"
#include "sixtracklib/cuda/context.h"

namespace SIXTRL_CXX_NAMESPACE
{
    CudaArgument::CudaArgument( CudaContext* SIXTRL_RESTRICT ctx ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase( ctx )
    {

    }

    CudaArgument::CudaArgument(
        CudaArgument::buffer_t const& SIXTRL_RESTRICT_REF buffer,
        CudaContext* SIXTRL_RESTRICT ctx )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        size_t const buffer_size = buffer.size();
        void const* buffer_data_begin = buffer.dataBegin< void const* >();

        if( ( buffer_size > size_t{ 0 } ) &&
            ( this->capacity() >= buffer_size ) &&
            ( this->cudaContext() == ctx ) &&
            ( ctx != nullptr ) && ( ctx->readyForSend() ) &&
            ( ctx->readyForRemap() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );

            status_t status = this->cudaContext()->send(
                this, buffer_data_begin, buffer_size );

            if( status == status_t{ 0 } )
            {
                status = ctx->remapSentCObjectsBuffer( this );
            }

            if( status == status_t{ 0 } )
            {
                this->doSetArgSize( buffer_size );
                this->doSetBufferRef( buffer );
            }
        }
    }

    CudaArgument::CudaArgument(
        const CudaArgument::c_buffer_t *const SIXTRL_RESTRICT ptr_c_buffer,
        CudaContext* SIXTRL_RESTRICT ctx ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase(
            ::NS(Buffer_get_size)( ptr_c_buffer ), ctx )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        size_t const buffer_size = ::NS(Buffer_get_size)( ptr_c_buffer );

        void const*  buffer_data_begin =
            ::NS(Buffer_get_const_data_begin)( ptr_c_buffer );

        if( ( buffer_size > size_t{ 0 } ) &&
            ( this->capacity() >= buffer_size ) &&
            ( this->cudaContext() == ctx ) &&
            ( ctx != nullptr ) && ( ctx->readyForSend() ) &&
            ( ctx->readyForRemap() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );

            status_t status = this->cudaContext()->send(
                this, buffer_data_begin, buffer_size );

            if( status == status_t{ 0 } )
            {
                status = ctx->remapSentCObjectsBuffer( this );
            }

            if( status == status_t{ 0 } )
            {
                this->doSetArgSize( buffer_size );
                this->doSetPtrCBuffer( ptr_c_buffer );
            }
        }
    }

    CudaArgument::CudaArgument( CudaArgument::size_type const capacity,
        CudaContext* SIXTRL_RESTRICT ctx ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase( capacity, ctx )
    {

    }

   CudaArgument::CudaArgument(
        const void *const SIXTRL_RESTRICT raw_arg_begin,
        CudaArgument::size_type const raw_arg_length,
        CudaContext* SIXTRL_RESTRICT ctx ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase( raw_arg_length, ctx )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        if( ( raw_arg_length > size_t{ 0 } ) &&
            ( this->cudaContext() != nullptr ) &&
            ( this->cudaContext() == ctx ) && ( ctx->readyForSend() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );
            SIXTRL_ASSERT( this->capacity() >= raw_arg_length );

            status_t status = this->cudaContext()->send(
                this, raw_arg_begin, raw_arg_length );

            if( status == status_t{ 0 } )
            {
                this->doSetArgSize( raw_arg_length );
                this->doSetPtrRawArgument( raw_arg_begin );
            }
        }
    }

    CudaArgument::ptr_cuda_context_t
    CudaArgument::cudaContext() SIXTRL_NOEXCEPT
    {
        using _this_t   = CudaArgument;
        using ptr_ctx_t = _this_t::ptr_cuda_context_t;

        return const_cast< ptr_ctx_t >( static_cast< _this_t const& >(
            *this ).cudaContext() );
    }

    CudaArgument::ptr_const_cuda_context_t
    CudaArgument::cudaContext() const SIXTRL_NOEXCEPT
    {
        using ptr_ctx_t = CudaArgument::ptr_const_cuda_context_t;
        ptr_ctx_t ptr_ctx = nullptr;

        if( ( this->ptrBaseContext() != nullptr ) &&
            ( this->ptrBaseContext()->type() == this->type() ) )
        {
            /* WARNING: * This down-casting is potentially dangerous (ub!) as
             * it relies on consistency of type() in both the argument and the
             * context! */

            ptr_ctx = static_cast< ptr_ctx_t >( this->ptrBaseContext() );
        }

        return ptr_ctx;
    }

}

/* end: sixtracklib/cuda/internal/argument.cu */
