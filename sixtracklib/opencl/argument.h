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
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/control/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

struct NS(Buffer);

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.hpp"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

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
    #endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */
#endif /* defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/cl.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    class ClContextBase;

    class ClArgument
    {
        public:

        using context_base_t     = ClContextBase;
        using size_type          = std::size_t;
        using cobj_buffer_t      = struct NS(Buffer);
        using cxx_cobj_buffer_t  = SIXTRL_CXX_NAMESPACE::Buffer;
        using status_t           = SIXTRL_CXX_NAMESPACE::arch_status_t;

        explicit ClArgument(
            context_base_t* SIXTRL_RESTRICT ptr_context = nullptr );

        explicit ClArgument(
            cxx_cobj_buffer_t& SIXTRL_RESTRICT_REF buffer,
            context_base_t* SIXTRL_RESTRICT ptr_context = nullptr );

        explicit ClArgument(
            cobj_buffer_t* SIXTRL_RESTRICT buffer,
            context_base_t* SIXTRL_RESTRICT ptr_context = nullptr );

        explicit ClArgument( size_type const arg_size,
            context_base_t* SIXTRL_RESTRICT ptr_context = nullptr );

        explicit ClArgument(
            void const* SIXTRL_RESTRICT arg_buffer_begin,
            size_type const arg_size,
            context_base_t* SIXTRL_RESTRICT ptr_context = nullptr );

        ClArgument( ClArgument const& orig ) = delete;
        ClArgument( ClArgument&& orig )      = delete;

        ClArgument& operator=( ClArgument const& orig ) = delete;
        ClArgument& operator=( ClArgument&& orig )      = delete;

        virtual ~ClArgument() SIXTRL_NOEXCEPT;

        size_type size() const SIXTRL_NOEXCEPT;

        bool write( cxx_cobj_buffer_t& SIXTRL_RESTRICT_REF buffer );
        bool write( cobj_buffer_t* SIXTRL_RESTRICT buffer );
        bool write( void const* SIXTRL_RESTRICT arg_buffer_begin,
                    size_type const arg_length );

        bool read(  cxx_cobj_buffer_t& SIXTRL_RESTRICT_REF buffer );
        bool read(  cobj_buffer_t* SIXTRL_RESTRICT_REF buffer );
        bool read(  void* SIXTRL_RESTRICT arg_buffer_begin,
                    size_type const arg_length );

        status_t updateRegion( size_type const offset, size_type length,
            void const* SIXTRL_RESTRICT new_value );

        status_t updateRegions( size_type const num_regions_to_update,
            size_type const* SIXTRL_RESTRICT offsets,
            size_type const* SIXTRL_RESTRICT lengths,
            void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values );

        bool usesCObjectBuffer() const SIXTRL_NOEXCEPT;
        cobj_buffer_t* ptrCObjectBuffer() const SIXTRL_NOEXCEPT;

        context_base_t* context() const SIXTRL_NOEXCEPT;

        bool attachTo(
            context_base_t* SIXTRL_RESTRICT ptr_context );

        cl::Buffer&         openClBuffer()       SIXTRL_NOEXCEPT;
        cl::Buffer const&   openClBuffer() const SIXTRL_NOEXCEPT;

        protected:

        virtual int doWriteAndRemapCObjBuffer(
            cobj_buffer_t* SIXTRL_RESTRICT_REF buffer );

        virtual int doReadAndRemapCObjBuffer(
            cobj_buffer_t* SIXTRL_RESTRICT buffer );

        virtual status_t doUpdateRegions( size_type const num_regions_to_update,
            size_type const* SIXTRL_RESTRICT offsets,
            size_type const* SIXTRL_RESTRICT lengths,
            void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values );

        void doSetCObjBuffer(
            cobj_buffer_t* SIXTRL_RESTRICT buffer ) const SIXTRL_NOEXCEPT;

        private:

        int doWriteAndRemapCObjBufferBaseImpl(
            cobj_buffer_t* SIXTRL_RESTRICT buffer );

        int doReadAndRemapCObjBufferBaseImpl(
            cobj_buffer_t* SIXTRL_RESTRICT buffer );

        cl::Buffer                      m_cl_buffer;

        mutable cobj_buffer_t*          m_ptr_cobj_buffer;
        mutable context_base_t*         m_ptr_context;
        size_type                       m_arg_size;
    };
}

typedef SIXTRL_CXX_NAMESPACE::ClArgument NS(ClArgument);

#else /* !defined( __cplusplus ) */

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

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/opencl/internal/base_context.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

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
    const NS(ClArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_HOST_FN bool NS(ClArgument_write)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_HOST_FN bool NS(ClArgument_write_memory)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    void const* SIXTRL_RESTRICT arg_buffer_begin,
    NS(context_size_t) const arg_length );

SIXTRL_HOST_FN bool NS(ClArgument_read)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(Buffer)* SIXTRL_RESTRICT buffer );

SIXTRL_HOST_FN bool NS(ClArgument_read_memory)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    void* arg_buffer_begin, NS(context_size_t) const arg_length );

SIXTRL_HOST_FN NS(arch_status_t) NS(ClArgument_update_region)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(context_size_t) const offset, NS(context_size_t) const length,
    void const* SIXTRL_RESTRICT new_value );

SIXTRL_HOST_FN NS(arch_status_t) NS(ClArgument_update_regions)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(context_size_t) const num_regions_to_update,
    NS(context_size_t) const* SIXTRL_RESTRICT offset,
    NS(context_size_t) const* SIXTRL_RESTRICT length,
    void const* SIXTRL_RESTRICT const* SIXTRL_RESTRICT new_values );

SIXTRL_HOST_FN bool NS(ClArgument_uses_cobj_buffer)(
    const NS(ClArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_HOST_FN NS(Buffer) const* NS(ClArgument_get_const_ptr_cobj_buffer)(
    const NS(ClArgument) *const SIXTRL_RESTRICT argument );

SIXTRL_HOST_FN NS(Buffer)* NS(ClArgument_get_ptr_cobj_buffer)(
    NS(ClArgument)* SIXTRL_RESTRICT argument );

SIXTRL_HOST_FN NS(ClContextBase)* NS(ClArgument_get_ptr_to_context)(
    NS(ClArgument)* SIXTRL_RESTRICT argument );

SIXTRL_HOST_FN bool NS(ClArgument_attach_to_context)(
    NS(ClArgument)* SIXTRL_RESTRICT argument,
    NS(ClContextBase)* SIXTRL_RESTRICT ptr_context );

SIXTRL_HOST_FN cl_mem NS(ClArgument_get_opencl_buffer)(
    NS(ClArgument)* SIXTRL_RESTRICT argument );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* !defined( __CUDACC__ ) */

#endif /* SIXTRACKLIB_OPENCL_ARGUMENT_H__ */

/* end: sixtracklib/opencl/argument.h */
