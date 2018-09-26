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
        using size_type      = std::size_t;

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

        bool write( struct NS(Buffer) const& buffer );
        bool write( void const* arg_buffer_begin, size_type const arg_length );

        bool read(  struct NS(Buffer)& buffer );
        bool read(  void* arg_buffer_begin, size_type const arg_length );

        context_base_t* context()       SIXTRL_NOEXCEPT;
        context_base_t const* context() const SIXTRL_NOEXCEPT;

        bool attachTo( ClContextBase* ptr_context );

        protected:

        virtual int doWriteAndRemapCObjBuffer( struct NS(Buffer) const& buffer );
        virtual int doReadAndRemapCObjBuffer(  struct NS(Buffer)& buffer );

        private:

        using raw_data_buffer_t = std::unique_ptr< unsigned char[] >;

        int doWriteAndRemapCObjBufferBaseImpl( struct NS(Buffer) const& buffer );
        int doReadAndRemapCObjBufferBaseImpl(  struct NS(Buffer)& buffer );

        cl::Buffer          m_cl_buffer;
        cl::Buffer          m_cl_success_flag;
        size_type           m_arg_size;
        context_base_t*     m_ptr_context;
    };
}

#else /* !defined( __cplusplus ) */

    #if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
        #include <CL/cl.h>
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

    #if !defined( SIXTRL_NO_INCLUDES )
        #include "sixtracklib/common/buffer.h"
    #endif /* !defined( SIXTRL_NO_INCLUDES ) */

#endif /* defined( __cplusplus ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */



#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_ARGUMENT_H__ */

/* end: sixtracklib/opencl/argument.h */
