#include "sixtracklib/opencl/argument.h"

#if !defined( __CUDACC__ )

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iterator>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/opencl/cl.h"
#include "sixtracklib/opencl/context.h"

namespace SIXTRL_CXX_NAMESPACE
{
    ClArgument::ClArgument(
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( ClArgument::size_type{ 0 } )
    {

    }

    ClArgument::ClArgument(
        ClArgument::cxx_cobj_buffer_t& SIXTRL_RESTRICT_REF buffer,
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( ClArgument::size_type{ 0 } )
    {
        size_type const arg_size = NS(Buffer_get_size)( buffer.getCApiPtr() );

        cl::Context* ptr_ocl_ctx = ( ptr_context != nullptr )
            ? ptr_context->openClContext() : nullptr;

        if( ( arg_size > size_type{ 0 } ) &&
            ( ptr_ocl_ctx != nullptr ) && ( ptr_context != nullptr ) &&
            ( ptr_context->openClQueue() != nullptr ) &&
            ( ptr_context->hasSelectedNode() ) &&
            ( ptr_context->hasRemappingKernel() ) )
        {
            this->m_cl_buffer = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, arg_size, nullptr );

            this->m_arg_size = arg_size;

            int const ret = this->doWriteAndRemapCObjBufferBaseImpl(
                buffer.getCApiPtr() );

            SIXTRL_ASSERT( ret == 0 );
            ( void )ret;
        }
    }

    ClArgument::ClArgument(
        ClArgument::cobj_buffer_t* SIXTRL_RESTRICT buffer,
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( ClArgument::size_type{ 0 } )
    {
        size_type const arg_size = NS(Buffer_get_size)( buffer );

        cl::Context* ptr_ocl_ctx = ( ptr_context != nullptr )
            ? ptr_context->openClContext() : nullptr;

        if( ( arg_size > size_type{ 0 } ) &&
            ( ptr_ocl_ctx != nullptr ) && ( ptr_context != nullptr ) &&
            ( ptr_context->openClQueue() != nullptr ) &&
            ( ptr_context->hasSelectedNode() ) &&
            ( ptr_context->hasRemappingKernel() ) )
        {
            this->m_cl_buffer = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, arg_size, nullptr );

            this->m_arg_size = arg_size;

            int const ret = this->doWriteAndRemapCObjBufferBaseImpl( buffer );

            SIXTRL_ASSERT( ret == 0 );
            ( void )ret;
        }
    }

    ClArgument::ClArgument(
        ClArgument::size_type const arg_size,
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( ClArgument::size_type{ 0 } )
    {
        cl::Context* ptr_ocl_ctx = ( ptr_context != nullptr )
            ? ptr_context->openClContext() : nullptr;

        cl::CommandQueue* ptr_ocl_queue = ( ptr_context != nullptr )
            ? ptr_context->openClQueue() : nullptr;

        if( ( arg_size > size_type{ 0 } ) &&
            ( ptr_ocl_ctx != nullptr ) && ( ptr_ocl_queue != nullptr ) &&
            ( ptr_context != nullptr ) &&
            ( ptr_context->hasSelectedNode() ) &&
            ( ptr_context->hasRemappingKernel() ) )
        {
            this->m_cl_buffer = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, arg_size, nullptr );

            this->m_arg_size = arg_size;
        }
    }

    ClArgument::ClArgument(
        void const* SIXTRL_RESTRICT arg_buffer_begin,
        ClArgument::size_type const arg_size,
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( ClArgument::size_type{ 0 } )
    {
        cl::Context* ptr_ocl_ctx = ( ptr_context != nullptr )
            ? ptr_context->openClContext() : nullptr;

        cl::CommandQueue* ptr_ocl_queue = ( ptr_context != nullptr )
            ? ptr_context->openClQueue() : nullptr;

        if( ( arg_buffer_begin != nullptr ) && ( arg_size > size_type{ 0 } ) &&
            ( ptr_ocl_ctx != nullptr ) && ( ptr_ocl_queue != nullptr ) &&
            ( ptr_context != nullptr ) &&
            ( ptr_context->hasSelectedNode() ) &&
            ( ptr_context->hasRemappingKernel() ) )
        {
            this->m_cl_buffer = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, arg_size, nullptr );

            this->m_arg_size = arg_size;

            cl_int cl_ret = ptr_ocl_queue->enqueueWriteBuffer(
                this->m_cl_buffer, CL_TRUE, 0, arg_size, arg_buffer_begin );

            SIXTRL_ASSERT( cl_ret == CL_SUCCESS );
            ( void )cl_ret;
        }
    }

    ClArgument::~ClArgument() SIXTRL_NOEXCEPT
    {

    }

    ClArgument::size_type ClArgument::size() const SIXTRL_NOEXCEPT
    {
        return this->m_arg_size;
    }

    bool ClArgument::write(
         ClArgument::cxx_cobj_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return ( 0 == this->doWriteAndRemapCObjBuffer( buffer.getCApiPtr() ) );
    }


    bool ClArgument::write( ClArgument::cobj_buffer_t* SIXTRL_RESTRICT buffer )
    {
        return ( 0 == this->doWriteAndRemapCObjBuffer( buffer ) );
    }

    bool ClArgument::write( void const* SIXTRL_RESTRICT arg_buffer_begin,
                            ClArgument::size_type const arg_length )
    {
        bool success = false;

        cl::CommandQueue* ptr_queue = ( this->m_ptr_context != nullptr )
            ? this->m_ptr_context->openClQueue() : nullptr;

        if( ( arg_buffer_begin != nullptr    ) &&
            ( arg_length > size_type{ 0 }    ) &&
            ( arg_length == this->m_arg_size ) &&
            ( ptr_queue != nullptr ) &&
            ( this->m_ptr_context != nullptr ) &&
            ( this->m_ptr_context->hasSelectedNode() ) )
        {
            cl_int ret = ptr_queue->enqueueWriteBuffer( this->m_cl_buffer,
                CL_TRUE, 0, arg_length, arg_buffer_begin );

            success = ( ret == CL_SUCCESS );
        }

        return success;
    }

    bool ClArgument::read(
        void* arg_buffer_begin, ClArgument::size_type const arg_length )
    {
        bool success = false;

        cl::CommandQueue* ptr_queue = ( this->m_ptr_context != nullptr )
            ? this->m_ptr_context->openClQueue() : nullptr;

        if( ( arg_buffer_begin != nullptr    ) &&
            ( arg_length > size_type{ 0 }    ) &&
            ( arg_length == this->m_arg_size ) &&
            ( ptr_queue != nullptr ) &&
            ( this->m_ptr_context != nullptr ) &&
            ( this->m_ptr_context->hasSelectedNode() ) )
        {
            cl_int ret = ptr_queue->enqueueReadBuffer( this->m_cl_buffer,
                CL_TRUE, 0, arg_length, arg_buffer_begin );

            success = ( ret == CL_SUCCESS );
        }

        return success;
    }

    bool ClArgument::read(
        ClArgument::cxx_cobj_buffer_t& SIXTRL_RESTRICT_REF buffer )
    {
        return ( 0 == this->doReadAndRemapCObjBuffer( buffer.getCApiPtr() ) );
    }

    bool ClArgument::read(
        ClArgument::cobj_buffer_t* SIXTRL_RESTRICT buffer )
    {
        return ( 0 == this->doReadAndRemapCObjBuffer( buffer ) );
    }

    ClArgument::status_t ClArgument::updateRegion(
        ClArgument::size_type const offset, ClArgument::size_type const length,
        void const* SIXTRL_RESTRICT new_value )
    {
        return this->doUpdateRegions( ClArgument::size_type{ 1 },
            &offset, &length, &new_value );
    }

    ClArgument::status_t ClArgument::updateRegions(
        ClArgument::size_type const num_regions_to_update,
        ClArgument::size_type const* SIXTRL_RESTRICT offsets,
        ClArgument::size_type const* SIXTRL_RESTRICT lengths,
        void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values )
    {
        return this->doUpdateRegions(
            num_regions_to_update, offsets, lengths, new_values );
    }

    bool ClArgument::usesCObjectBuffer() const SIXTRL_NOEXCEPT
    {
        return ( this->m_ptr_cobj_buffer != nullptr );
    }

    ClArgument::cobj_buffer_t*
    ClArgument::ptrCObjectBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_cobj_buffer;
    }


    ClArgument::context_base_t*
    ClArgument::context() const SIXTRL_NOEXCEPT
    {
        return this->m_ptr_context;
    }

    bool ClArgument::attachTo(
         ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context )
    {
        bool success = false;

        if( ( this->m_ptr_context != ptr_context ) &&
            ( ptr_context != nullptr ) &&
            ( this->m_cl_buffer.getInfo< CL_MEM_SIZE >() == size_type{ 0 } ) )
        {
            this->m_ptr_context = ptr_context;
            success = true;
        }

        return success;
    }

    cl::Buffer&  ClArgument::openClBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_cl_buffer;
    }

    cl::Buffer const& ClArgument::openClBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_cl_buffer;
    }

    int ClArgument::doWriteAndRemapCObjBuffer(
        ClArgument::cobj_buffer_t* SIXTRL_RESTRICT buffer )
    {
        return this->doWriteAndRemapCObjBufferBaseImpl( buffer );
    }

    int ClArgument::doReadAndRemapCObjBuffer(
        ClArgument::cobj_buffer_t* SIXTRL_RESTRICT buffer )
    {
        return this->doReadAndRemapCObjBufferBaseImpl( buffer );
    }

    ClArgument::status_t ClArgument::doUpdateRegions(
        ClArgument::size_type const num_regions_to_update,
        ClArgument::size_type const* SIXTRL_RESTRICT offsets,
        ClArgument::size_type const* SIXTRL_RESTRICT lengths,
        void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values )
    {
        using size_t = ClArgument::size_type;

        ClArgument::status_t status = ::NS(ARCH_STATUS_GENERAL_FAILURE);
        cl::CommandQueue* ptr_queue = ( this->m_ptr_context != nullptr )
            ? this->m_ptr_context->openClQueue() : nullptr;

        size_t const arg_size = this->m_arg_size;

        if( ( offsets != nullptr ) && ( lengths != nullptr ) &&
            ( new_values != nullptr ) && ( arg_size > size_t{ 0 } ) &&
            ( num_regions_to_update > ClArgument::size_type{ 0 } ) &&
            ( ptr_queue != nullptr ) &&
            ( this->m_ptr_context != nullptr ) &&
            ( this->m_ptr_context->hasSelectedNode() ) )
        {
            status = ::NS(ARCH_STATUS_SUCCESS);

            for( size_t ii = size_t{ 0 }; ii < num_regions_to_update ; ++ii )
            {
                size_t const offset = offsets[ ii ];
                size_t const length = lengths[ ii ];
                void const* SIXTRL_RESTRICT new_value = new_values[ ii ];

                if( ( new_value == nullptr ) || ( length == size_t{ 0 } ) ||
                    ( ( offset + length ) > arg_size ) )
                {
                    status = ::NS(ARCH_STATUS_GENERAL_FAILURE);
                    break;
                }

                cl_int ret = ptr_queue->enqueueWriteBuffer(
                    this->m_cl_buffer, CL_TRUE, offset, length, new_value );

                if( ret != CL_SUCCESS )
                {
                    status = ::NS(ARCH_STATUS_GENERAL_FAILURE);
                    break;
                }
            }
        }

        return status;
    }

    int ClArgument::doWriteAndRemapCObjBufferBaseImpl(
        ClArgument::cobj_buffer_t* SIXTRL_RESTRICT buffer )
    {
        int success = -1;

        size_type const buffer_size = NS(Buffer_get_size)( buffer );

        cl::CommandQueue* ptr_queue = ( this->m_ptr_context != nullptr )
            ? this->m_ptr_context->openClQueue() : nullptr;

        if( ( this->m_ptr_context != nullptr ) &&
            ( buffer != nullptr ) && ( ptr_queue != nullptr ) &&
            ( this->m_ptr_context->hasSelectedNode() ) &&
            ( this->m_ptr_context->hasRemappingKernel() ) &&
            ( buffer_size > size_type{ 0 } ) &&
            ( buffer_size <= this->m_arg_size ) )
        {
            ClContextBase::kernel_id_t remap_kernel_id =
                this->m_ptr_context->remappingKernelId();

            SIXTRL_ASSERT( this->m_ptr_context->openClKernel( remap_kernel_id )
                != nullptr );

            size_type const num_args =
                this->m_ptr_context->kernelNumArgs( remap_kernel_id );

            if( num_args > size_type{ 0 } )
            {
                success    = 0;
                cl_int ret = CL_SUCCESS ;

                if( num_args > size_type{ 1 } )
                {
                    int32_t success_flag = 0;
                    success = ( num_args == size_type{ 2 } ) ? 0 : -1;

                    if( success == 0 )
                    {
                        ret = ptr_queue->enqueueWriteBuffer(
                            this->m_ptr_context->internalSuccessFlagBuffer(),
                            CL_TRUE, 0, sizeof( success_flag ), &success_flag );

                        success = ( ret == CL_SUCCESS ) ? 0 : -2;
                    }

                    if( success == 0 )
                    {
                        this->m_ptr_context->assignKernelArgumentClBuffer(
                            remap_kernel_id, 1u,
                            this->m_ptr_context->internalSuccessFlagBuffer() );
                    }
                }

                if( ( success == 0 ) && ( num_args > size_type{ 0 } ) )
                {
                    ret = ptr_queue->enqueueWriteBuffer(
                        this->m_cl_buffer, CL_TRUE, 0, buffer_size,
                            NS(Buffer_get_const_data_begin)( buffer ) );

                    if( ret == CL_SUCCESS )
                    {
                        this->m_ptr_context->assignKernelArgumentClBuffer(
                            remap_kernel_id, 0u, this->m_cl_buffer );
                    }
                    else
                    {
                        success = -4;
                    }
                }

                if( success == 0 )
                {
                    size_type const num_worker_items =
                        this->m_ptr_context->kernelPreferredWorkGroupSizeMultiple(
                            remap_kernel_id );

                    success = ( this->m_ptr_context->runKernel(
                        remap_kernel_id, num_worker_items ) ) ? 0 : -8;
                }

                if( ( success == 0 ) && ( num_args > size_type{ 1 } ) )
                {
                    int32_t success_flag = int32_t{ 0 };

                    ret = ptr_queue->enqueueReadBuffer(
                        this->m_ptr_context->internalSuccessFlagBuffer(),
                        CL_TRUE, 0, sizeof( success_flag ), &success_flag );

                    success = ( ret == CL_SUCCESS ) ? 0 : -2;

                    if( success == 0 )
                    {
                        success |= success_flag;
                    }
                }

                if( success == 0 )
                {
                    ptr_queue->flush();
                }

                if( success == 0 )
                {
                    this->doSetCObjBuffer( buffer );
                }
            }
        }

        return success;
    }

    int ClArgument::doReadAndRemapCObjBufferBaseImpl(
        ClArgument::cobj_buffer_t * SIXTRL_RESTRICT buffer )
    {
        int success = -1;

        size_type const buffer_capacity = NS(Buffer_get_capacity)( buffer );

        cl::CommandQueue* ptr_queue = ( this->m_ptr_context != nullptr )
            ? this->m_ptr_context->openClQueue() : nullptr;

        if( ( this->m_ptr_context != nullptr ) &&
            ( buffer != nullptr ) && ( ptr_queue != nullptr ) &&
            ( this->m_ptr_context->hasSelectedNode() ) &&
            ( this->m_arg_size >  size_type{ 0 } ) &&
            ( buffer_capacity >=  this->m_arg_size ) )
        {
            cl_int ret = ptr_queue->enqueueReadBuffer(
                this->m_cl_buffer, CL_TRUE, 0, this->m_arg_size,
                NS(Buffer_get_data_begin)( buffer ) );

            success = ( ret == CL_SUCCESS ) ? 0 : -1;

            if( success == 0 )
            {
                success = NS(Buffer_remap)( buffer );
            }

            if( success == 0 )
            {
                this->doSetCObjBuffer( buffer );
            }
        }

        return success;
    }

    void ClArgument::doSetCObjBuffer(
         ClArgument::cobj_buffer_t* SIXTRL_RESTRICT buffer ) const SIXTRL_NOEXCEPT
    {
        this->m_ptr_cobj_buffer = buffer;
        return;
    }
}

/* ========================================================================= */

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new)(
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context )
{
    return new SIXTRL_CXX_NAMESPACE::ClArgument( ptr_context );
}

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new_from_buffer)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context )
{
    return ( buffer != nullptr )
        ? new SIXTRL_CXX_NAMESPACE::ClArgument( buffer, ptr_context )
        : new SIXTRL_CXX_NAMESPACE::ClArgument( ptr_context );
}

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new_from_size)(
    NS(context_size_t) const arg_size,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context )
{
    return new SIXTRL_CXX_NAMESPACE::ClArgument( arg_size, ptr_context );
}

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new_from_memory)(
    void const* SIXTRL_RESTRICT arg_buffer_begin,
    NS(context_size_t) const arg_size,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context )
{
    return new SIXTRL_CXX_NAMESPACE::ClArgument(
        arg_buffer_begin, arg_size, ptr_context );
}

SIXTRL_HOST_FN void NS(ClArgument_delete)(
    NS(ClArgument)* SIXTRL_RESTRICT argument )
{
    delete argument;
}

SIXTRL_HOST_FN NS(context_size_t) NS(ClArgument_get_argument_size)(
    const NS(ClArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr )
        ? argument->size() : NS(context_size_t){ 0 };
}

SIXTRL_HOST_FN bool NS(ClArgument_write)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return ( ( argument != nullptr ) && ( buffer != nullptr ) )
        ? argument->write( buffer ) : false;
}

SIXTRL_HOST_FN bool NS(ClArgument_write_memory)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    void const* SIXTRL_RESTRICT arg_buffer_begin,
    NS(context_size_t) const arg_length )
{
    return ( argument != nullptr )
        ? argument->write( arg_buffer_begin, arg_length ) : false;
}

SIXTRL_HOST_FN bool NS(ClArgument_read)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(Buffer)* SIXTRL_RESTRICT buffer )
{
    return ( ( argument != nullptr ) && ( buffer != nullptr ) )
        ? argument->read( buffer ) : false;
}

SIXTRL_HOST_FN bool NS(ClArgument_read_memory)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    void* SIXTRL_RESTRICT arg_buffer_begin,
    NS(context_size_t) const arg_length )
{
    return ( argument != nullptr )
        ? argument->read( arg_buffer_begin, arg_length ) : false;
}

SIXTRL_HOST_FN NS(arch_status_t) NS(ClArgument_update_region)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(context_size_t) const offset, NS(context_size_t) const length,
    void const* SIXTRL_RESTRICT new_value )
{
    return ( argument != nullptr )
        ? argument->updateRegion( offset, length, new_value )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

SIXTRL_HOST_FN NS(arch_status_t) NS(ClArgument_update_regions)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(context_size_t) const num_regions_to_update,
    NS(context_size_t) const* SIXTRL_RESTRICT offsets,
    NS(context_size_t) const* SIXTRL_RESTRICT lengths,
    void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values )
{
    return ( argument != nullptr ) ? argument->updateRegions(
        num_regions_to_update, offsets, lengths, new_values )
        : ::NS(ARCH_STATUS_GENERAL_FAILURE);
}

SIXTRL_HOST_FN bool NS(ClArgument_uses_cobj_buffer)(
    const NS(ClArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->usesCObjectBuffer() : false;
}

SIXTRL_HOST_FN NS(Buffer) const* NS(ClArgument_get_const_ptr_cobj_buffer)(
    const NS(ClArgument) *const SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrCObjectBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(Buffer)* NS(ClArgument_get_ptr_cobj_buffer)(
    NS(ClArgument)* SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->ptrCObjectBuffer() : nullptr;
}

SIXTRL_HOST_FN NS(ClContextBase)* NS(ClArgument_get_ptr_to_context)(
    NS(ClArgument)* SIXTRL_RESTRICT argument )
{
    return ( argument != nullptr ) ? argument->context() : nullptr;
}

SIXTRL_HOST_FN bool NS(ClArgument_attach_to_context)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context )
{
    return ( argument != nullptr ) ? argument->attachTo( ptr_context ) : false;
}

SIXTRL_HOST_FN cl_mem NS(ClArgument_get_opencl_buffer)(
    NS(ClArgument)* SIXTRL_RESTRICT argument )
{
    cl::Buffer* ptr_arg_buffer = ( argument != nullptr )
        ? &argument->openClBuffer() : nullptr;

    if( ptr_arg_buffer != nullptr )
    {
        return ptr_arg_buffer->operator()();
    }

    return cl_mem{};
}

#endif /* !defined( __CUDACC__ ) */

/* end: sixtracklib/opencl/code/argument.cpp */
