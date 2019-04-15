#include "sixtracklib/cuda/internal/context_base.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/context/definitions.h"
#include "sixtracklib/common/context/context_base.h"
#include "sixtracklib/common/context/context_base_with_nodes.h"
#include "sixtracklib/common/buffer.h"

#include "sixtracklib/cuda/internal/argument_base.h"
#include "sixtracklib/cuda/wrappers/context_operations.h"

namespace SIXTRL_CXX_NAMESPACE
{
    CudaContextBase::CudaContextBase( char const* config_str ) :
        ContextOnNodesBase( SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_CUDA,
            SIXTRL_CONTEXT_TYPE_CUDA_STR, config_str )
    {

    }

    CudaContextBase::status_t CudaContextBase::doSend(
        CudaContextBase::ptr_arg_base_t SIXTRL_RESTRICT dest_arg,
        const void *const SIXTRL_RESTRICT source,
        CudaContextBase::size_type const source_length )
    {
        using _this_t    = CudaContextBase;
        using status_t   = _this_t::status_t;
        using size_t     = _this_t::size_type;

        status_t status = status_t{ -1 };

        SIXTRL_ASSERT( this->readyForSend() );
        SIXTRL_ASSERT( dest_arg != nullptr );
        SIXTRL_ASSERT( source   != nullptr );
        SIXTRL_ASSERT( source_length > size_t{ 0 } );
        SIXTRL_ASSERT( dest_arg->capacity() >= source_length );

        if( ( dest_arg->hasArgumentBuffer() ) &&
            ( this->type() == dest_arg->type() ) )
        {
            using cuda_arg_t = SIXTRL_CXX_NAMESPACE::CudaArgumentBase;
            using cuda_arg_buffer_t = cuda_arg_t::cuda_arg_buffer_t;

            /* WARNING: * This down-casting is potentially dangerous (ub!) as
             * it relies on consistency of the type() information for both
             * context and * argument! */

            cuda_arg_t* cuda_arg = static_cast< cuda_arg_t* >( dest_arg );

            if( cuda_arg->hasCudaArgBuffer() )
            {
                cuda_arg_buffer_t arg_buffer = cuda_arg->cudaArgBuffer();
                SIXTRL_ASSERT( arg_buffer != nullptr );

                status = ::NS(CudaContext_perform_send)(
                    arg_buffer, source, source_length );
            }
        }

        return status;
    }

    CudaContextBase::status_t CudaContextBase::doReceive(
        void* SIXTRL_RESTRICT destination,
        CudaContextBase::size_type const dest_capacity,
        CudaContextBase::ptr_arg_base_t SIXTRL_RESTRICT src_arg )
    {
        using _this_t    = CudaContextBase;
        using status_t   = _this_t::status_t;
        using size_t     = _this_t::size_type;

        status_t status = status_t{ -1 };

        SIXTRL_ASSERT( this->readyForReceive() );
        SIXTRL_ASSERT( destination != nullptr );
        SIXTRL_ASSERT( src_arg != nullptr );
        SIXTRL_ASSERT( src_arg->size() > size_t{ 0 } );
        SIXTRL_ASSERT( dest_capacity >= src_arg->size() );

        if( ( src_arg->hasArgumentBuffer() ) &&
            ( this->type() == src_arg->type() ) )
        {
            using cuda_arg_t = SIXTRL_CXX_NAMESPACE::CudaArgumentBase;
            using cuda_arg_buffer_t = cuda_arg_t::cuda_arg_buffer_t;

            /* WARNING: * This down-casting is potentially dangerous (ub!) as
             * it relies on consistency of the type() information for both
             * context and * argument! */

            cuda_arg_t* cuda_arg = static_cast< cuda_arg_t* >( src_arg );

            if( cuda_arg->hasCudaArgBuffer() )
            {
                cuda_arg_buffer_t arg_buffer = cuda_arg->cudaArgBuffer();
                SIXTRL_ASSERT( arg_buffer != nullptr );

                status = ::NS(CudaContext_perform_receive)( destination,
                    dest_capacity, arg_buffer, src_arg->size() );
            }
        }

        return status;
    }

    CudaContextBase::status_t CudaContextBase::doRemapSentCObjectsBuffer(
        CudaContextBase::ptr_arg_base_t SIXTRL_RESTRICT arg )
    {
        using  _this_t = CudaContextBase;
        using status_t = _this_t::status_t;
        using   size_t = _this_t::size_type;

        status_t status = status_t{ -1 };

        SIXTRL_ASSERT( this->readyForRemap() );
        SIXTRL_ASSERT( arg != nullptr );
        SIXTRL_ASSERT( arg->size() > size_t{ 0 } );

        if( ( arg->hasArgumentBuffer() ) && ( this->type() == arg->type() ) &&
            ( arg->usesCObjectsBuffer() ) )
        {
            using cuda_arg_t = SIXTRL_CXX_NAMESPACE::CudaArgumentBase;
            using c_buffer_t = cuda_arg_t::c_buffer_t;
            using cuda_arg_buffer_t = cuda_arg_t::cuda_arg_buffer_t;

            /* WARNING: * This down-casting is potentially dangerous (ub!) as
             * it relies on consistency of the type() information for both
             * context and * argument! */

            cuda_arg_t* cuda_arg = static_cast< cuda_arg_t* >( arg );

            if( cuda_arg->hasCudaArgBuffer() )
            {
                cuda_arg_buffer_t arg_buffer = cuda_arg->cudaArgBuffer();
                SIXTRL_ASSERT( arg_buffer != nullptr );

                c_buffer_t* buffer = cuda_arg->ptrCObjectsBuffer();
                SIXTRL_ASSERT( buffer != nullptr );

                size_t const slot_size = ::NS(Buffer_get_slot_size)( buffer );
                status = ::NS(CudaContext_perform_remap_send_cobject_buffer)(
                    arg_buffer, slot_size );
            }
        }

        return status;
    }
}

/* end: sixtracklib/cuda/internal/context_base.cpp */
