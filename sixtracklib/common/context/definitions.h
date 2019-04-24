#ifndef SIXTRACKLIB_COMMON_CONTEXT_DEFINITIONS_H__
#define SIXTRACKLIB_COMMON_CONTEXT_DEFINITIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host */

typedef SIXTRL_UINT64_T  NS(context_type_id_t);
typedef SIXTRL_INT32_T   NS(context_status_t);
typedef SIXTRL_UINT64_T  NS(context_size_t);
typedef SIXTRL_UINT64_T  NS(context_success_flag_t);

/* Predefined  context type id's: limit them to 0x0000 - 0x01FF */
/* For userdefined type id's, the range 0x0200 - 0x03FF is reserved */

#if !defined( SIXTRL_CONTEXT_TYPE_ID_BITMASK )
    #define SIXTRL_CONTEXT_TYPE_ID_BITMASK 0x000003FF
#endif /* !defined( SIXTRL_CONTEXT_TYPE_ID_BITMASK) */

#if !defined( SIXTRL_CONTEXT_TYPE_ID_OFFSET )
    #define SIXTRL_CONTEXT_TYPE_ID_OFFSET 0
#endif /* !defined( SIXTRL_CONTEXT_TYPE_ID_OFFSET) */

/* Then reserve 8 bits to encode up to 256 variants for any specific
 * type_id */

#if !defined( SIXTRL_CONTEXT_TYPE_VARIANT_ID_BITMASK )
    #define SIXTRL_CONTEXT_TYPE_VARIANT_ID_BITMASK 0x000FF000
#endif /* !defined( SIXTRL_CONTEXT_TYPE_VARIANT_ID_BITMASK ) */

#if !defined( SIXTRL_CONTEXT_TYPE_VARIANT_ID_OFFSET )
    #define SIXTRL_CONTEXT_TYPE_VARIANT_ID_OFFSET 10
#endif /* !defined( SIXTRL_CONTEXT_TYPE_VARIANT_ID_OFFSET ) */

/* ------------------------------------------------------------------------ */

#if !defined( SIXTRL_CONTEXT_TYPE_INVALID)
    #define SIXTRL_CONTEXT_TYPE_INVALID 0x000003FF
#endif /* !defined( SIXTRL_CONTEXT_TYPE_INVALID) */

#if !defined( SIXTRL_CONTEXT_TYPE_NONE)
    #define SIXTRL_CONTEXT_TYPE_NONE 0x00000000
#endif /* !defined( SIXTRL_CONTEXT_TYPE_NONE) */

#if !defined( SIXTRL_CONTEXT_TYPE_CPU)
    #define SIXTRL_CONTEXT_TYPE_CPU 0x00000001
#endif /* !defined( SIXTRL_CONTEXT_TYPE_CPU) */

#if !defined(SIXTRL_CONTEXT_TYPE_CPU_STR)
    #define SIXTRL_CONTEXT_TYPE_CPU_STR "cpu"
#endif /* !defined(SIXTRL_CONTEXT_TYPE_CPU_STR) */

#if !defined( SIXTRL_CONTEXT_TYPE_OPENCL)
    #define SIXTRL_CONTEXT_TYPE_OPENCL 0x00000002
#endif /* !defined( SIXTRL_CONTEXT_TYPE_OPENCL) */

#if !defined(SIXTRL_CONTEXT_TYPE_OPENCL_STR)
    #define SIXTRL_CONTEXT_TYPE_OPENCL_STR "opencl"
#endif /* !defined(SIXTRL_CONTEXT_TYPE_OPENCL_STR) */

#if !defined( SIXTRL_CONTEXT_TYPE_CUDA)
    #define SIXTRL_CONTEXT_TYPE_CUDA 0x00000003
#endif /* !defined( SIXTRL_CONTEXT_TYPE_CUDA) */

#if !defined(SIXTRL_CONTEXT_TYPE_CUDA_STR)
    #define SIXTRL_CONTEXT_TYPE_CUDA_STR "cuda"
#endif /* !defined(SIXTRL_CONTEXT_TYPE_CUDA_STR) */

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )

SIXTRL_STATIC_VAR NS(context_type_id_t) const NS(CONTEXT_TYPE_ID_BITMASK) =
    ( NS(context_type_id_t) )SIXTRL_CONTEXT_TYPE_ID_BITMASK;

SIXTRL_STATIC_VAR NS(context_type_id_t) const NS(CONTEXT_TYPE_ID_OFFSET) =
    ( NS(context_type_id_t) )SIXTRL_CONTEXT_TYPE_ID_OFFSET;

SIXTRL_STATIC_VAR NS(context_type_id_t) const
    NS(CONTEXT_TYPE_VARIANT_ID_BITMASK) = ( NS(context_type_id_t)
        )SIXTRL_CONTEXT_TYPE_VARIANT_ID_BITMASK;

SIXTRL_STATIC_VAR NS(context_type_id_t) const
    NS(CONTEXT_TYPE_VARIANT_ID_OFFSET) = ( NS(context_type_id_t)
        )SIXTRL_CONTEXT_TYPE_VARIANT_ID_OFFSET;

SIXTRL_STATIC_VAR NS(context_type_id_t) const NS(CONTEXT_TYPE_INVALID) =
    ( NS(context_type_id_t) )SIXTRL_CONTEXT_TYPE_INVALID;

SIXTRL_STATIC_VAR NS(context_type_id_t) const NS(CONTEXT_TYPE_NONE) =
    ( NS(context_type_id_t) )SIXTRL_CONTEXT_TYPE_NONE;

SIXTRL_STATIC_VAR NS(context_type_id_t) const NS(CONTEXT_TYPE_CPU) =
    ( NS(context_type_id_t) )SIXTRL_CONTEXT_TYPE_CPU;

SIXTRL_STATIC_VAR NS(context_type_id_t) const NS(CONTEXT_TYPE_OPENCL) =
    ( NS(context_type_id_t) )SIXTRL_CONTEXT_TYPE_OPENCL;

SIXTRL_STATIC_VAR NS(context_type_id_t) const NS(CONTEXT_TYPE_CUDA) =
    ( NS(context_type_id_t) )SIXTRL_CONTEXT_TYPE_CUDA;

#endif /* !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}

namespace SIXTRL_CXX_NAMESPACE
{
    using context_type_id_t      = SIXTRL_UINT64_T;
    using context_status_t       = SIXTRL_INT32_T;
    using context_success_flag_t = SIXTRL_UINT64_T;
    using context_size_t         = SIXTRL_UINT64_T;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_ID_BITMASK = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_ID_BITMASK );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_ID_OFFSET = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_ID_OFFSET );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_VARIANT_ID_BITMASK = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_VARIANT_ID_BITMASK );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_VARIANT_ID_OFFSET = static_cast< context_type_id_t
            >( SIXTRL_CONTEXT_TYPE_VARIANT_ID_OFFSET );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_INVALID = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_INVALID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_NONE = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_NONE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_CPU = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_CPU );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_OPENCL = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_OPENCL );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_type_id_t
        CONTEXT_TYPE_CUDA = static_cast< context_type_id_t >(
            SIXTRL_CONTEXT_TYPE_CUDA );
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_DEFINITIONS_H__ */

/* end: sixtracklib/common/context/definitions.h */