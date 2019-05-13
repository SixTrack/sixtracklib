#include "sixtracklib/cuda/argument.hpp"

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.hpp"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/common/control/definitions.h"
#include "sixtracklib/common/control/argument_base.hpp"
#include "sixtracklib/common/control/controller_base.hpp"

#include "sixtracklib/cuda/definitions.h"
#include "sixtracklib/cuda/control/argument_base.hpp"
#include "sixtracklib/cuda/controller.hpp"

namespace SIXTRL_CXX_NAMESPACE
{
    CudaArgument::CudaArgument( CudaController* SIXTRL_RESTRICT ctrl ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase( ctrl )
    {

    }

    CudaArgument::CudaArgument(
        CudaArgument::buffer_t const& SIXTRL_RESTRICT_REF buffer,
        CudaController* SIXTRL_RESTRICT ctrl )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        size_t const buffer_size = buffer.size();
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

            status_t status = this->cudaController()->send(
                this, buffer_data_begin, buffer_size );

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                status = ctrl->remapCObjectsBuffer( this );
            }

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                this->doSetArgSize( buffer_size );
                this->doSetBufferRef( buffer );
            }
        }
    }

    CudaArgument::CudaArgument(
        const CudaArgument::c_buffer_t *const SIXTRL_RESTRICT ptr_c_buffer,
        CudaController* SIXTRL_RESTRICT ctrl ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase(
            ::NS(Buffer_get_size)( ptr_c_buffer ), ctrl )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        size_t const buffer_size = ::NS(Buffer_get_size)( ptr_c_buffer );

        void const*  buffer_data_begin =
            ::NS(Buffer_get_const_data_begin)( ptr_c_buffer );

        if( ( buffer_size > size_t{ 0 } ) &&
            ( this->capacity() >= buffer_size ) &&
            ( this->cudaController() == ctrl ) &&
            ( ctrl != nullptr ) && ( ctrl->readyForSend() ) &&
            ( ctrl->readyForRemap() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );

            status_t status = this->cudaController()->send(
                this, buffer_data_begin, buffer_size );

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                status = ctrl->remapCObjectsBuffer( this );
            }

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                this->doSetArgSize( buffer_size );
                this->doSetPtrCBuffer( ptr_c_buffer );
            }
        }
    }

    CudaArgument::CudaArgument( CudaArgument::size_type const capacity,
        CudaController* SIXTRL_RESTRICT ctrl ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase( capacity, ctrl )
    {

    }

   CudaArgument::CudaArgument(
        const void *const SIXTRL_RESTRICT raw_arg_begin,
        CudaArgument::size_type const raw_arg_length,
        CudaController* SIXTRL_RESTRICT ctrl ) :
        SIXTRL_CXX_NAMESPACE::CudaArgumentBase( raw_arg_length, ctrl )
    {
        using status_t = CudaArgument::status_t;
        using size_t   = CudaArgument::size_type;

        if( ( raw_arg_length > size_t{ 0 } ) &&
            ( this->cudaController() != nullptr ) &&
            ( this->cudaController() == ctrl ) && ( ctrl->readyForSend() ) )
        {
            SIXTRL_ASSERT( this->hasCudaArgBuffer() );
            SIXTRL_ASSERT( this->cudaArgBuffer() != nullptr );
            SIXTRL_ASSERT( this->size() == size_t{ 0 } );
            SIXTRL_ASSERT( this->capacity() >= raw_arg_length );

            status_t status = this->cudaController()->send(
                this, raw_arg_begin, raw_arg_length );

            if( status == ::NS(ARCH_STATUS_SUCCESS) )
            {
                this->doSetArgSize( raw_arg_length );
                this->doSetPtrRawArgument( raw_arg_begin );
            }
        }
    }

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
        using cuda_ctrl_t = SIXTRL_CXX_NAMESPACE::CudaController;

        return ( this->ptrControllerBase() != nullptr )
            ? this->ptrControllerBase()->asDerivedController< cuda_ctrl_t >(
                    SIXTRL_CXX_NAMESPACE::ARCHITECTURE_CUDA ) : nullptr;
    }
}

#endif /* C++, Host */

/* end: sixtracklib/cuda/control/argument.cpp */
