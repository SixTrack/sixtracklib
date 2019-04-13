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

namespace SIXTRL_CXX_NAMESPACE
{
    CudaArgumentBase::~CudaArgumentBase() SIXTRL_NOEXCEPT
    {
        this->doDeleteCudaArgumentBuffer();
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        ArgumentBase( true, ptr_context ),
        m_arg_buffer( nullptr )
    {

    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::buffer_t& SIXTRL_RESTRICT_REF buffer,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        ArgumentBase( true, ptr_context ),
        m_arg_buffer( nullptr )
    {
        if( 0 == this->doInitWriteBufferCudaBaseImpl( buffer.getCApiPtr() ) )
        {
            this->doSetPtrC99Buffer( buffer.getCApiPtr() );
            this->doSetPtrCxxBuffer( &buffer );
        }
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::c_buffer_t* SITRL_RESTRICT ptr_c_buffer,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        ArgumentBase( true, ptr_context ),
        m_arg_buffer( nullptr )
    {
        if( 0 == this->doInitWriteBufferCudaBaseImpl( ptr_c_buffer ) )
        {
            this->doSetPtrC99Buffer( ptr_c_buffer );
        }
    }

    CudaArgumentBase::CudaArgumentBase(
        CudaArgumentBase::size_type const arg_size,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context ) :
        ArgumentBase( true, ptr_context ),
        m_arg_buffer( nullptr )
    {
        using size_t = CudaArgumentBase::size_type;

        if( ( arg_size > size_t{ 0 } ) &&
            ( 0 == this->doReserveArgumentBufferCudaBaseImpl( arg_size ) ) )
        {
            this->doSetArgSize( arg_size );
        }
    }

    CudaArgumentBase::CudaArgumentBase(
        void const* SIXTRL_RESTRICT raw_argument_begin,
        CudaArgumentBase::size_type const arg_size,
        CudaArgumentBase::ptr_context_t SIXTRL_RESTRICT ptr_context )  :
        ArgumentBase( true, ptr_context ),
        m_arg_buffer( nullptr )
    {
        using size_t = CudaArgumentBase::size_type;

        if( raw_argument_begin != nullptr ) && ( arg_size > size_t{ 0 } )
        {
            bool succes = false;

            if( 0 == this->doReserveArgumentBufferCudaBaseImpl( arg_size ) )
            {
                success = ( 0 == this->doTransferBufferToDevice(
                    raw_argument_begin, arg_size ) );
            }

            if( success )
            {
                this->doSetPtrRawArgument( raw_argument_begin );
                this->doSetArgSize( arg_size );
            }
        }
    }

    CudaArgumentBase::ptr_cuda_arg_buffer_t
    CudaArgumentBase::doGetCudaArgumentBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_arg_buffer;
    }

    CudaArgumentBase::ptr_const_cuda_arg_buffer_t
    CudaArgumentBase::doGetCudaArgumentBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_buffer;
    }

    void CudaArgumentBase::doDeleteCudaArgumentBuffer() SIXTRL_NOEXCEPT
    {
        using size_t = CudaArgumentBase::size_type;

        if( this->m_arg_buffer != nullptr )
        {
            delete this->m_arg_buffer;
            this->m_arg_buffer = nullptr;

            this->doSetArgSize( size_t{ 0 } );
            this->doSetArgCapacity( size_t{ 0 } );
        }

        return;
    }

    void CudaArgumentBase::doResetCudaArgumentBuffer(
        CudaArgumentBase::ptr_cuda_arg_buffer_t SIXTRL_RESTRICT new_arg_buffer,
        CudaArgumentBase::size_type const capacity )
    {
        using size_t = CudaArgumentBase::size_type;

        if( new_arg_buffer != this->doGetCudaArgumentBuffer() )
        {
            this->doDeleteCudaArgumentBuffer();

            this->doSetPtrC99Buffer( nullptr );
            this->doSetPtrCxxBuffer( nullptr );
            this->doSetPtrRawArgument( nullptr );

            if( ( new_arg_buffer != nullptr ) &&
                ( capacity > size_t{ 0 } ) )
            {
                this->m_arg_buffer = new_arg_buffer;
                this->doSetArgCapacity( capacity );
            }
        }

        return;
    }

    /* ----------------------------------------------------------------- */

    int CudaArgumentBase::doReserveArgumentBuffer(
        CudaArgumentBase::size_type const requ_buffer_size )
    {
        return this->doReserveArgumentBufferCudaBaseImpl( requ_buffer_size );
    }

    int CudaArgumentBase::doTransferBufferToDevice(
        void const* SIXTRL_RESTRICT source_begin,
        CudaArgumentBase::size_type const buffer_size )
    {
        return this->doTransferBufferToDeviceCudaBaseImpl(
            source_begin, buffer_size );
    }

    int CudaArgumentBase::doTransferBufferFromDevice(
        void* SIXTRL_RESTRICT dest_begin,
        CudaArgumentBase::size_type const buffer_size )
    {
        return this->doTransferBufferFromDeviceCudaBaseImpl(
            dest_begin, buffer_size );
    }

    int CudaArgumentBase::doRemapCObjectBufferAtDevice()
    {
        return this->doRemapCObjectBufferAtDeviceCudaBaseImpl();
    }

    int CudaArgumentBase::doInitWriteBufferCudaBaseImpl(
        CudaArgumentBase::c_buffer_t* SIXTRL_RESTRICT ptr_c_buffer )
    {

    }

    int CudaArgumentBase::doReserveArgumentBufferCudaBaseImpl(
        CudaArgumentBase::size_type const required_buffer_size )
    {

    }

    int CudaArgumentBase::doTransferBufferToDeviceCudaBaseImpl(
        void const* SIXTRL_RESTRICT source_buffer_begin,
        CudaArgumentBase::size_type const buffer_size )
    {

    }

    int CudaArgumentBase::doTransferBufferFromDeviceCudaBaseImpl(
        void* SIXTRL_RESTRICT dest_buffer_begin,
        CudaArgumentBase::size_type const buffer_size )
    {

    }

    int CudaArgumentBase::doRemapCObjectBufferAtDeviceCudaBaseImpl()
    {

    }
}

/* end: sixtracklib/cuda/internal/argument_base.cu */
