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

::NS(CudaArgument)* NS(CudaArgument_new)(
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( ctx );
}

::NS(CudaArgument)*
NS(CudaArgument_new_from_buffer)( ::NS(Buffer)* SIXTRL_RESTRICT buffer,
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( buffer, ctx );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_size)(
    ::NS(context_size_t) const capacity,
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument( capacity, ctx );
}

::NS(CudaArgument)* NS(CudaArgument_new_from_memory)(
    void const* SIXTRL_RESTRICT raw_arg_begin,
    ::NS(context_size_t) const raw_arg_length,
    ::NS(CudaContext)* SIXTRL_RESTRICT ctx )
{
    return new SIXTRL_CXX_NAMESPACE::CudaArgument(
        raw_arg_begin, raw_arg_length, ctx );
}

void NS(CudaArgument_delete)( ::NS(CudaArgument)* SIXTRL_RESTRICT argument )
{
    delete argument;
}

bool NS(CudaArgument_send_buffer)( ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    const NS(Buffer) *const SIXTRL_RESTRICT source_buffer )
{
    return ( arg != nullptr ) ? arg->send( source_buffer ) : false;
}

bool NS(CudaArgument_send_memory)( ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    const void *const SIXTRL_RESTRICT source_arg_begin,
    ::NS(context_size_t) const source_arg_length )
{
    return ( arg != nullptr )
        ? arg->send( source_arg_begin, source_arg_length ) : false;
}

bool NS(CudaArgument_receive_buffer)( ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    ::NS(Buffer)* SIXTRL_RESTRICT destination_buffer )
{
    return ( arg != nullptr ) ? arg->receive( destination_buffer ) : false;
}

bool NS(CudaArgument_receive_memory)( ::NS(CudaArgument)* SIXTRL_RESTRICT arg,
    void* SIXTRL_RESTRICT destination_buffer,
    ::NS(context_size_t) const destination_capacity )
{
    return ( arg != nullptr )
        ? arg->receive( destination_buffer, arg->size() ) : false;
}


bool NS(CudaArgument_uses_cobjects_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->usesCObjectsBuffer() : false;
}

NS(Buffer)* NS(CudaArgument_get_cobjects_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrCObjectsBuffer() : nullptr;
}

bool NS(CudaArgument_uses_raw_argument)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->usesRawArgument() : false;
}

bool NS(CudaArgument_get_ptr_raw_argument)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrRawArgument() : nullptr;
}

NS(context_size_t) NS(CudaArgument_get_size)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr )
        ? argument->size() : ::NS(context_size_t){ 0 };
}

NS(context_size_t) NS(CudaArgument_get_capacity)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr )
        ? argument->capacity() : ::NS(context_size_t){ 0 };
}

bool NS(CudaArgument_has_argument_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->hasArgumentBuffer() : false;
}

bool NS(CudaArgument_requires_argument_buffer)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->requiresArgumentBuffer() : false;
}

NS(context_type_id_t) NS(CudaArgument_get_type_id)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->type()
        : SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_INVALID;
}

char const* NS(CudaArgument_get_type_id_str)(
    const NS(CudaArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrTypeStr() : nullptr;
}

/* end: sixtracklib/cuda/internal/argument.cu */
