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

#include <CL/cl.hpp>

#include "sixtracklib/common/definitions.h"
#include "sixtracklib/common/buffer.h"
#include "sixtracklib/opencl/context.h"

namespace SIXTRL_CXX_NAMESPACE
{
    ClArgument::ClArgument(
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_cl_success_flag(),
        m_ptr_cobj_buffer( nullptr ),
        m_ptr_context( ptr_context ),
        m_arg_size( ClArgument::size_type{ 0 } )
    {

    }

    ClArgument::ClArgument(
        ClArgument::cxx_cobj_buffer_t& SIXTRL_RESTRICT_REF buffer,
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_cl_success_flag(),
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

            this->m_cl_success_flag = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, sizeof( int32_t ), nullptr );

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
        m_cl_success_flag(),
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

            this->m_cl_success_flag = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, sizeof( int32_t ), nullptr );

            int const ret = this->doWriteAndRemapCObjBufferBaseImpl( buffer );

            SIXTRL_ASSERT( ret == 0 );
            ( void )ret;
        }
    }

    ClArgument::ClArgument(
        ClArgument::size_type const arg_size,
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_cl_success_flag(),
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

            this->m_cl_success_flag = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, sizeof( int32_t ), nullptr );

            this->m_arg_size = arg_size;
        }
    }

    ClArgument::ClArgument(
        void const* SIXTRL_RESTRICT arg_buffer_begin,
        ClArgument::size_type const arg_size,
        ClArgument::context_base_t* SIXTRL_RESTRICT ptr_context ) :
        m_cl_buffer(),
        m_cl_success_flag(),
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

            this->m_cl_success_flag = cl::Buffer(
                *ptr_ocl_ctx, CL_MEM_READ_WRITE, sizeof( int32_t ), nullptr );

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

    cl::Buffer const& ClArgument::internalSuccessFlagBuffer() const SIXTRL_NOEXCEPT
    {
        return this->m_cl_success_flag;
    }

    cl::Buffer& ClArgument::internalSuccessFlagBuffer() SIXTRL_NOEXCEPT
    {
        return this->m_cl_success_flag;
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

            cl::Kernel* ptr_remapping_kernel =
                this->m_ptr_context->openClKernel( remap_kernel_id );

            SIXTRL_ASSERT( ptr_remapping_kernel != nullptr );

            size_type const num_args =
                this->m_ptr_context->kernelNumArgs( remap_kernel_id );

            size_type const work_group_prefered_multiple =
                this->m_ptr_context->kernelPreferredWorkGroupSizeMultiple(
                    remap_kernel_id );

            if( ( work_group_prefered_multiple > size_type{ 0 } ) &&
                ( num_args > size_type{ 0 } ) )
            {
                cl_int ret = CL_SUCCESS ;

                if( num_args > size_type{ 1 } )
                {
                    int32_t success_flag = 0;
                    success = ( num_args == size_type{ 2 } ) ? 0 : -1;

                    if( success == 0 )
                    {
                        ret = ptr_queue->enqueueWriteBuffer(
                            this->m_cl_success_flag, CL_TRUE, 0,
                            sizeof( success_flag ), &success_flag );

                        success = ( ret == CL_SUCCESS ) ? 0 : -2;
                    }

                    if( success == 0 )
                    {
                        ret = ptr_remapping_kernel->setArg(
                            1, this->m_cl_success_flag );

                        success = ( CL_SUCCESS == ret ) ? 0 : -4;
                    }
                }
                else
                {
                    success = 0;
                }

                ret = ptr_queue->enqueueWriteBuffer(
                    this->m_cl_buffer, CL_TRUE, 0, buffer_size,
                    NS(Buffer_get_const_data_begin)( buffer ) );

                success = ( ret == CL_SUCCESS ) ? 0 : -2;

                if( success == 0 )
                {
                    ret = ptr_remapping_kernel->setArg( 0, this->m_cl_buffer );
                    success = ( CL_SUCCESS == ret ) ? 0 : -4;
                }

                if( success == 0 )
                {
                    ret = ptr_queue->enqueueNDRangeKernel(
                        *ptr_remapping_kernel, cl::NullRange,
                        cl::NDRange( work_group_prefered_multiple ),
                        cl::NDRange( work_group_prefered_multiple ) );

                    success = ( ret == CL_SUCCESS ) ? 0 : -8;
                }

                if( ( success == 0 ) && ( num_args > size_type{ 1 } ) )
                {
                    int32_t success_flag = int32_t{ 0 };

                    ret = ptr_queue->enqueueReadBuffer( this->m_cl_success_flag,
                        CL_TRUE, 0, sizeof( success_flag ), &success_flag );

                    success = ( ret == CL_SUCCESS ) ? 0 : -2;

                    if( success == 0 )
                    {
                        success |= success_flag;
                    }
                }
                else if( success == 0 )
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
    const NS(ClArgument) *const SIXTRL_RESTRICT argument )
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

SIXTRL_HOST_FN cl_mem NS(ClArgument_get_internal_opencl_success_flag_buffer)(
    NS(ClArgument)* SIXTRL_RESTRICT argument )
{
    cl::Buffer* ptr_success_flag = ( argument != nullptr ) ?
        &argument->internalSuccessFlagBuffer() : nullptr;

    if( ptr_success_flag != nullptr )
    {
        return ptr_success_flag->operator()();
    }

    return cl_mem{};
}

#endif /* !defined( __CUDACC__ ) */

/* end: sixtracklib/opencl/code/argument.cpp */
