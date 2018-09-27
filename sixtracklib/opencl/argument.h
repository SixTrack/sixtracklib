#ifndef SIXTRACKLIB_OPENCL_ARGUMENT_H__
#define SIXTRACKLIB_OPENCL_ARGUMENT_H__

#if !defined( __CUDACC__ )

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <limits.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

struct NS(Buffer);

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if defined( __cplusplus )

    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <iterator>
        #include <memory>
        #include <string>
        #include <map>
        #include <vector>

        #include <CL/cl.hpp>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

namespace SIXTRL_NAMESPACE
{
    class ClContextBase;

    class ClArgument
    {
        public:

        using context_base_t = ClContextBase;
        using size_type      = ClContextBase::size_type;
        using cobj_buffer_t  = Buffer;

        explicit ClArgument( struct NS(Buffer) const& buffer,
            context_base_t* ptr_context = nullptr );

        explicit ClArgument( size_type const arg_size,
            context_base_t* ptr_context = nullptr );

        explicit ClArgument(
            void const* arg_buffer_begin, size_type const arg_size,
            context_base_t* ptr_context = nullptr );

        ClArgument( ClArgument const& orig ) = delete;
        ClArgument( ClArgument&& orig )      = delete;

        ClArgument& operator=( ClArgument const& orig ) = delete;
        ClArgument& operator=( ClArgument&& orig )      = delete;

        virtual ~ClArgument() SIXTRL_NOEXCEPT;

        size_type size() const SIXTRL_NOEXCEPT;

        bool write( Buffer const& buffer );
        bool write( void const* arg_buffer_begin, size_type const arg_length );

        bool read(  Buffer& buffer );
        bool read(  void* arg_buffer_begin, size_type const arg_length );

        bool isCObjectBufferArgument() const SIXTRL_NOEXCEPT;
        Buffer* ptrCObjectBuffer() const SIXTRL_NOEXCEPT;

        context_base_t* context()       SIXTRL_NOEXCEPT;
        context_base_t const* context() const SIXTRL_NOEXCEPT;

        bool attachTo( ClContextBase* ptr_context );

        cl::Buffer&         openClBuffer()       SIXTRL_NOEXCEPT;
        cl::Buffer const&   openClBuffer() const SIXTRL_NOEXCEPT;

        cl::Buffer const& internalSuccessFlagBuffer() const SIXTRL_NOEXCEPT;
        cl::Buffer& internalSuccessFlagBuffer() SIXTRL_NOEXCEPT;

        protected:

        virtual int doWriteAndRemapCObjBuffer( struct NS(Buffer) const& buffer );
        virtual int doReadAndRemapCObjBuffer(  struct NS(Buffer)& buffer );

        void doSetCObjBuffer( Buffer& buffer ) SIXTRL_NOEXCEPT;
        void doSetCObjBuffer( Buffer const& buffer ) SIXTRL_NOEXCEPT;

        private:

        int doWriteAndRemapCObjBufferBaseImpl( struct NS(Buffer) const& buffer );
        int doReadAndRemapCObjBufferBaseImpl(  struct NS(Buffer)& buffer );

        cl::Buffer          m_cl_buffer;
        cl::Buffer          m_cl_success_flag;

        mutable Buffer*     m_ptr_cobj_buffer;
        size_type           m_arg_size;
        context_base_t*     m_ptr_context;
    };
}

typedef SIXTRL_NAMESPACE::ClArgument NS(ClArgument);

#else /* !defined( __cplusplus ) */

    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <CL/cl.h>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef void NS(ClArgument);

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new)(
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context );

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new_from_buffer)(
    NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context );

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new_from_size)(
    NS(context_size_t) const arg_size,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context );

SIXTRL_HOST_FN NS(ClArgument)* NS(ClArgument_new_from_memory)(
    void const* SIXTRL_RESTRICT arg_buffer_begin,
    NS(context_size_t) const arg_size,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context );

SIXTRL_HOST_FN void NS(ClArgument_delete)(
    NS(ClArgument)* SIXTRL_RESTRICT argument );

SIXTRL_HOST_FN NS(context_size_t) NS(ClArgument_get_argument_size)(


SIXTRL_HOST_FN bool NS(ClArgument_write)( struct NS(Buffer) const& buffer );
SIXTRL_HOST_FN bool NS(ClArgument_write_memory)( void const* arg_buffer_begin, size_type const arg_length );

SIXTRL_HOST_FN bool read(  struct NS(Buffer)& buffer );
SIXTRL_HOST_FN bool read(  void* arg_buffer_begin, size_type const arg_length );

SIXTRL_HOST_FN bool isCObjectBufferArgument() const SIXTRL_NOEXCEPT;
SIXTRL_HOST_FN Buffer* ptrCObjectBuffer() const SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN context_base_t* context()       SIXTRL_NOEXCEPT;
SIXTRL_HOST_FN context_base_t const* context() const SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN bool attachTo( ClContextBase* ptr_context );

SIXTRL_HOST_FN cl::Buffer&         openClBuffer()       SIXTRL_NOEXCEPT;
SIXTRL_HOST_FN cl::Buffer const&   openClBuffer() const SIXTRL_NOEXCEPT;

SIXTRL_HOST_FN cl::Buffer const& internalSuccessFlagBuffer() const SIXTRL_NOEXCEPT;
SIXTRL_HOST_FN cl::Buffer& internalSuccessFlagBuffer() SIXTRL_NOEXCEPT;



#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_ARGUMENT_H__ */

/* end: sixtracklib/opencl/argument.h */
