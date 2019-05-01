#ifndef SIXTRACKLIB_COMMON_CONTEXT_DEFINITIONS_H__
#define SIXTRACKLIB_COMMON_CONTEXT_DEFINITIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

typedef SIXTRL_UINT64_T  NS(context_type_id_t);
typedef SIXTRL_INT32_T   NS(context_status_t);
typedef SIXTRL_UINT64_T  NS(context_size_t);
typedef SIXTRL_UINT64_T  NS(context_success_flag_t);

typedef SIXTRL_UINT64_T  NS(arch_id_t);
typedef SIXTRL_UINT64_T  NS(arch_size_t);

typedef SIXTRL_INT64_T   NS(node_platform_id_t);
typedef SIXTRL_INT64_T   NS(node_device_id_t);
typedef SIXTRL_UINT32_T  NS(node_index_t);

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

#if !defined(SIXTRL_CONTEXT_STATUS_SUCCESS)
    #define SIXTRL_CONTEXT_STATUS_SUCCESS 0
#endif /* !defined(SIXTRL_CONTEXT_STATUS_SUCCESS) */

#if !defined(SIXTRL_CONTEXT_STATUS_GENERAL_FAILURE)
    #define SIXTRL_CONTEXT_STATUS_GENERAL_FAILURE -1
#endif /* !defined(SIXTRL_CONTEXT_STATUS_GENERAL_FAILURE) */

#if !defined( _GPUCODE )

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

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC_VAR NS(context_status_t) const NS(CONTEXT_STATUS_SUCCESS) =
    ( NS(context_status_t) )SIXTRL_CONTEXT_STATUS_SUCCESS;

SIXTRL_STATIC_VAR NS(context_status_t) const
    NS(CONTEXT_STATUS_GENERAL_FAILURE) =
        ( NS(context_status_t) )SIXTRL_CONTEXT_STATUS_GENERAL_FAILURE;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC_VAR NS(node_platform_id_t) const NS(NODE_ILLEGAL_PATFORM_ID) =
    ( NS(node_platform_id_t) )-1;

SIXTRL_STATIC_VAR NS(node_device_id_t) const NS(NODE_ILLEGAL_DEVICE_ID) =
    ( NS(node_device_id_t) )-1;

SIXTRL_STATIC_VAR NS(node_index_t) const NS(NODE_UNDEFINED_INDEX) =
    ( NS(node_index_t) )0xFFFFFFFF;


#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}

namespace SIXTRL_CXX_NAMESPACE
{
    typedef ::NS(context_type_id_t)         context_type_id_t;
    typedef ::NS(context_status_t)          context_status_t;
    typedef ::NS(context_success_flag_t)    context_success_flag_t;
    typedef ::NS(context_size_t)            context_size_t;

    typedef ::NS(arch_id_t)                 arch_id_t;
    typedef ::NS(arch_size_t)               arch_size_t;

    typedef ::NS(node_platform_id_t)        node_platform_id_t;
    typedef ::NS(node_device_id_t)          node_device_id_t;
    typedef ::NS(node_index_t)              node_index_t;

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

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_status_t
        CONTEXT_STATUS_SUCCESS = static_cast< context_status_t >(
            SIXTRL_CONTEXT_STATUS_SUCCESS );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST context_status_t
        CONTEXT_STATUS_GENERAL_FAILURE = static_cast< context_status_t >(
            SIXTRL_CONTEXT_STATUS_GENERAL_FAILURE );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_platform_id_t
        NODE_ILLEGAL_PATFORM_ID = static_cast< node_platform_id_t >( -1 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_device_id_t
        NODE_ILLEGAL_DEVIVE_ID = static_cast< node_device_id_t <( -1 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_index_t
        NODE_UNDEFINED_INDEX = static_cast< node_index_t >( 0xFFFFFFFF );
}

#endif /* C++, Host */

#endif /* SIXTRACKLIB_COMMON_CONTEXT_DEFINITIONS_H__ */

/* end: sixtracklib/common/context/definitions.h */
