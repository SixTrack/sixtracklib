#ifndef SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_ABS_BASE_H__
#define SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_ABS_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/generated/namespace.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef SIXTRL_INT32_T NS(context_type_int_t);

typedef enum NS(context_type_t)
{
    NS(CONTEXT_TYPE_INVALID)            = ( NS(context_type_int_t) )-1,
    NS(CONTEXT_TYPE_NONE)               = ( NS(context_type_int_t) )0x00000000,
    NS(CONTEXT_TYPE_BEGIN)              = ( NS(context_type_int_t) )0x00000001,
    NS(CONTEXT_TYPE_CPU_SINGLE)         = ( NS(context_type_int_t) )0x00000001,
    NS(CONTEXT_TYPE_OPENCL)             = ( NS(context_type_int_t) )0x00000002,
    NS(CONTEXT_TYPE_CUDA)               = ( NS(context_type_int_t) )0x00000003,
    NS(CONTEXT_TYPE_END)                = ( NS(context_type_int_t) )0x00000003
}
NS(context_type_t);

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if defined( __cplusplus )

namespace SIXTRL_CXX_NAMESPACE
{
    using context_type_int_t = ::NS(context_type_int_t);

    using context_type_t =
    enum context_type_enum_e : context_type_int_t
    {
        CONTEXT_TYPE_INVALID =
            static_cast< context_type_int_t >( ::NS(CONTEXT_TYPE_INVALID) ),

        CONTEXT_TYPE_NONE =
            static_cast< context_type_int_t >( ::NS(CONTEXT_TYPE_NONE) ),

        CONTEXT_TYPE_CPU_SINGLE =
            static_cast< context_type_int_t >( ::NS(CONTEXT_TYPE_CPU_SINGLE) ),

        CONTEXT_TYPE_OPENCL =
            static_cast< context_type_int_t >( ::NS(CONTEXT_TYPE_OPENCL) ),

        CONTEXT_TYPE_CUDA =
            static_cast< context_type_int_t >( ::NS(CONTEXT_TYPE_CUDA) )
    };

    class ContextBase
    {
        public:

        using type_id_t = context_type_t;
        using size_type = uint64_t;

        type_id_t type() const SIXTRL_NOEXCEPT;

        virtual ~ContextBase() = default;

        protected:

        using type_id_int_t = context_type_int_t;

        explicit ContextBase( type_id_t const type =
            SIXTRL_CXX_NAMESPACE::CONTEXT_TYPE_INVALID ) SIXTRL_NOEXCEPT;

        ContextBase( ContextBase const& other ) = default;
        ContextBase( ContextBase&& other ) = default;

        ContextBase& operator=( ContextBase const& rhs ) = default;
        ContextBase& operator=( ContextBase&& rhs ) = default;

        private:

        type_id_t   m_type_id;
    };
}

#if !defined( _GPUCODE )
extern "C" {
#endif /* !defined( _GPUCODE ) */

typedef sixtrack::ContextBase               NS(ContextBase);
typedef sixtrack::ContextBase::size_type    NS(context_size_t);

#if !defined( _GPUCODE )
}
#endif /* !defined( _GPUCODE ) */

#else /* !defined( __cplusplus ) */

typedef void                NS(ContextBase);
typedef SIXTRL_UINT64_T     NS(context_size_t);

#endif /* defined( __cplusplus ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

SIXTRL_EXTERN SIXTRL_HOST_FN NS(context_type_t) NS(ContextBase_get_type)(
    const NS(ContextBase) *const SIXTRL_RESTRICT context );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */


#endif /* SIXTRACKLIB_COMMON_CONTEXT_CONTEXT_ABS_BASE_H__ */

/* end: sixtracklib/common/context/context_abs_base.h */
