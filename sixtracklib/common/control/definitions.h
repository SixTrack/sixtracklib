#ifndef SIXTRACKLIB_COMMON_CONTROL_DEFINITIONS_H__
#define SIXTRACKLIB_COMMON_CONTROL_DEFINITIONS_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, Host */

typedef SIXTRL_UINT64_T       NS(arch_id_t);
typedef SIXTRL_INT32_T        NS(arch_status_t);
typedef SIXTRL_UINT64_T       NS(arch_size_t);
typedef SIXTRL_UINT64_T       NS(arch_debugging_t);
typedef SIXTRL_UINT32_T       NS(arch_kernel_id_t);
typedef SIXTRL_UINT32_T       NS(arch_program_id_t);

typedef NS(arch_status_t)     NS(ctrl_status_t);
typedef NS(arch_size_t)       NS(ctrl_size_t);
typedef NS(arch_kernel_id_t)  NS(ctrl_kernel_id_t);
typedef NS(arch_program_id_t) NS(ctrl_program_id_t);

typedef SIXTRL_INT64_T        NS(node_platform_id_t);
typedef SIXTRL_INT64_T        NS(node_device_id_t);
typedef SIXTRL_UINT32_T       NS(node_index_t);
typedef SIXTRL_UINT16_T       NS(node_id_str_fmt_t);

typedef enum NS(ctrl_perform_remap_flag_e)
{
    NS(CTRL_PERFORM_REMAP) = 0,
    NS(CTRL_PERFORM_NO_REMAP) = 1
}
NS(ctrl_perform_remap_flag_t);

/* Predefined  architecture type id's: limit them to 0x0000 - 0x01FF */
/* For userdefined type id's, the range 0x0200 - 0x03FF is reserved */

#if !defined( SIXTRL_ARCHITECTURE_ID_BITMASK )
    #define SIXTRL_ARCHITECTURE_ID_BITMASK 0x000003FF
#endif /* !defined( SIXTRL_ARCHITECTURE_ID_BITMASK) */

#if !defined( SIXTRL_ARCHITECTURE_ID_OFFSET )
    #define SIXTRL_ARCHITECTURE_ID_OFFSET 0
#endif /* !defined( SIXTRL_ARCHITECTURE_ID_OFFSET) */

/* Then reserve 8 bits to encode up to 256 variants for any specific
 * type_id */

#if !defined( SIXTRL_ARCHITECTURE_ID_VARIANT_BITMASK )
    #define SIXTRL_ARCHITECTURE_ID_VARIANT_BITMASK 0x000FF000
#endif /* !defined( SIXTRL_ARCHITECTURE_ID_VARIANT_BITMASK ) */

#if !defined( SIXTRL_ARCHITECTURE_ID_VARIANT_OFFSET )
    #define SIXTRL_ARCHITECTURE_ID_VARIANT_OFFSET 10
#endif /* !defined( SIXTRL_ARCHITECTURE_ID_VARIANT_OFFSET ) */

/* ------------------------------------------------------------------------ */

#if !defined( SIXTRL_ARCHITECTURE_ILLEGAL)
    #define SIXTRL_ARCHITECTURE_ILLEGAL 0x000003FF
#endif /* !defined( SIXTRL_ARCHITECTURE_ILLEGAL) */

#if !defined( SIXTRL_ARCHITECTURE_NONE)
    #define SIXTRL_ARCHITECTURE_NONE 0x00000000
#endif /* !defined( SIXTRL_ARCHITECTURE_NONE) */

#if !defined( SIXTRL_ARCHITECTURE_CPU)
    #define SIXTRL_ARCHITECTURE_CPU 0x00000001
#endif /* !defined( SIXTRL_ARCHITECTURE_CPU) */

#if !defined(SIXTRL_ARCHITECTURE_CPU_STR)
    #define SIXTRL_ARCHITECTURE_CPU_STR "cpu"
#endif /* !defined(SIXTRL_ARCHITECTURE_CPU_STR) */

#if !defined( SIXTRL_ARCHITECTURE_OPENCL)
    #define SIXTRL_ARCHITECTURE_OPENCL 0x00000002
#endif /* !defined( SIXTRL_ARCHITECTURE_OPENCL) */

#if !defined(SIXTRL_ARCHITECTURE_OPENCL_STR)
    #define SIXTRL_ARCHITECTURE_OPENCL_STR "opencl"
#endif /* !defined(SIXTRL_ARCHITECTURE_OPENCL_STR) */

#if !defined( SIXTRL_ARCHITECTURE_CUDA)
    #define SIXTRL_ARCHITECTURE_CUDA 0x00000003
#endif /* !defined( SIXTRL_ARCHITECTURE_CUDA) */

#if !defined(SIXTRL_ARCHITECTURE_CUDA_STR)
    #define SIXTRL_ARCHITECTURE_CUDA_STR "cuda"
#endif /* !defined(SIXTRL_ARCHITECTURE_CUDA_STR) */

/* ------------------------------------------------------------------------- */

#if !defined(SIXTRL_ARCH_STATUS_SUCCESS)
    #define SIXTRL_ARCH_STATUS_SUCCESS 0
#endif /* !defined(SIXTRL_ARCH_STATUS_SUCCESS) */

#if !defined(SIXTRL_ARCH_STATUS_GENERAL_FAILURE)
    #define SIXTRL_ARCH_STATUS_GENERAL_FAILURE -1
#endif /* !defined(SIXTRL_ARCH_STATUS_GENERAL_FAILURE) */

/* ------------------------------------------------------------------------- */

#if !defined( SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY )
    #define SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY 0
#endif /* SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY */

#if !defined( SIXTRL_ARCH_DEBUGGING_MIN_FLAG )
    #define SIXTRL_ARCH_DEBUGGING_MIN_FLAG 0x0000000100000000
#endif /* !defined( SIXTRL_ARCH_DEBUGGING_MIN_FLAG ) */

#if !defined( SIXTRL_ARCH_DEBUGGING_MAX_FLAG )
    #define SIXTRL_ARCH_DEBUGGING_MAX_FLAG 0x8000000000000000
#endif /* !defined( SIXTRL_ARCH_DEBUGGING_MAX_FLAG ) */

#if !defined( SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK )
    #define SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK 0x00000000ffffffff
#endif /* !defined( SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK ) */

#if !defined( SIXTRL_ARCH_DEBUGGING_STATUS_INV_BITMASK )
    #define SIXTRL_ARCH_DEBUGGING_STATUS_INV_BITMASK 0xffffffff00000000
#endif /* !defined( SIXTRL_ARCH_DEBUGGING_STATUS_INV_BITMASK ) */

#if !defined( SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG )
    #define SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG 0x80000000
#endif /* !defined( SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG ) */

#if !defined( SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE )
    #define SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE 0xFFFFFFFF00000000
#endif /* !defined( SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE ) */

/* ------------------------------------------------------------------------- */

#if !defined( SIXTRL_ARCH_ILLEGAL_BUFFER_ID)
    #define SIXTRL_ARCH_ILLEGAL_BUFFER_ID 0
#endif /* !defined( SIXTRL_ARCH_ILLEGAL_BUFFER_ID) */

#if !defined( SIXTRL_ARCH_PARTICLES_BUFFER_ID )
    #define SIXTRL_ARCH_PARTICLES_BUFFER_ID 1
#endif /* !defined( SIXTRL_ARCH_PARTICLES_BUFFER_ID ) */

#if !defined( SIXTRL_ARCH_BEAM_ELEMENTS_BUFFER_ID )
    #define SIXTRL_ARCH_BEAM_ELEMENTS_BUFFER_ID 2
#endif /* !defined( SIXTRL_ARCH_BEAM_ELEMENTS_BUFFER_ID ) */

#if !defined( SIXTRL_ARCH_OUTPUT_BUFFER_ID )
    #define SIXTRL_ARCH_OUTPUT_BUFFER_ID 3
#endif /* !defined( SIXTRL_ARCH_OUTPUT_BUFFER_ID ) */

#if !defined( SIXTRL_ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID )
    #define SIXTRL_ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID 4
#endif /* !defined( SIXTRL_ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID ) */

#if !defined( SIXTRL_ARCH_PARTICLE_ADDR_BUFFER_ID )
    #define SIXTRL_ARCH_PARTICLE_ADDR_BUFFER_ID 5
#endif /* !defined( SIXTRL_ARCH_PARTICLE_ADDR_BUFFER_ID ) */

#if !defined( SIXTRL_ARCH_MIN_USER_DEFINED_BUFFER_ID )
    #define SIXTRL_ARCH_MIN_USER_DEFINED_BUFFER_ID 1024
#endif /* !defined( SIXTRL_ARCH_MIN_USER_DEFINED_BUFFER_ID ) */

#if !defined( SIXTRL_ARCH_EXTERNAL_BUFFER_ID )
    #define SIXTRL_ARCH_EXTERNAL_BUFFER_ID 1024
#endif /* !defined( SIXTRL_ARCH_EXTERNAL_BUFFER_ID ) */

#if !defined( SIXTRL_ARCH_MAX_USER_DEFINED_BUFFER_ID )
    #define SIXTRL_ARCH_MAX_USER_DEFINED_BUFFER_ID 0xFFFFFFFFFFFFFFFE
#endif /* !defined( SIXTRL_ARCH_MAX_USER_DEFINED_BUFFER_ID ) */

/* ------------------------------------------------------------------------- */

#if !defined( SIXTRL_NODE_ID_STR_FORMAT_NOARCH)
    #define SIXTRL_NODE_ID_STR_FORMAT_NOARCH 0
#endif /* !defined( SIXTRL_NODE_ID_STR_FORMAT_NOARCH) */

#if !defined( SIXTRL_NODE_ID_STR_FORMAT_ARCHID)
    #define SIXTRL_NODE_ID_STR_FORMAT_ARCHID 1
#endif /* !defined( SIXTRL_NODE_ID_STR_FORMAT_ARCHID) */

#if !defined( SIXTRL_NODE_ID_STR_FORMAT_ARCHSTR)
    #define SIXTRL_NODE_ID_STR_FORMAT_ARCHSTR 2
#endif /* !defined( SIXTRL_NODE_ID_STR_FORMAT_ARCHSTR) */

#if !defined( SIXTRL_NODE_ID_STR_FORMAT_DEFAULT)
    #define SIXTRL_NODE_ID_STR_FORMAT_DEFAULT 0
#endif /* !defined( SIXTRL_NODE_ID_STR_FORMAT_DEFAULT) */

#if !defined( SIXTRL_NODE_ID_STR_FORMAT_ILLEGAL)
    #define SIXTRL_NODE_ID_STR_FORMAT_ILLEGAL 0xffff
#endif /* !defined( SIXTRL_NODE_ID_STR_FORMAT_ILLEGAL) */

#if !defined( _GPUCODE )

SIXTRL_STATIC_VAR NS(arch_id_t) const NS(ARCHITECTURE_ID_BITMASK) =
    ( NS(arch_id_t) )SIXTRL_ARCHITECTURE_ID_BITMASK;

SIXTRL_STATIC_VAR NS(arch_id_t) const NS(ARCHITECTURE_ID_OFFSET) =
    ( NS(arch_id_t) )SIXTRL_ARCHITECTURE_ID_OFFSET;

SIXTRL_STATIC_VAR NS(arch_id_t) const
    NS(ARCHITECTURE_ID_VARIANT_BITMASK) = ( NS(arch_id_t)
        )SIXTRL_ARCHITECTURE_ID_VARIANT_BITMASK;

SIXTRL_STATIC_VAR NS(arch_id_t) const
    NS(ARCHITECTURE_ID_VARIANT_OFFSET) = ( NS(arch_id_t)
        )SIXTRL_ARCHITECTURE_ID_VARIANT_OFFSET;

SIXTRL_STATIC_VAR NS(arch_id_t) const NS(ARCHITECTURE_ILLEGAL) =
    ( NS(arch_id_t) )SIXTRL_ARCHITECTURE_ILLEGAL;

SIXTRL_STATIC_VAR NS(arch_id_t) const NS(ARCHITECTURE_NONE) =
    ( NS(arch_id_t) )SIXTRL_ARCHITECTURE_NONE;

SIXTRL_STATIC_VAR NS(arch_id_t) const NS(ARCHITECTURE_CPU) =
    ( NS(arch_id_t) )SIXTRL_ARCHITECTURE_CPU;

SIXTRL_STATIC_VAR NS(arch_id_t) const NS(ARCHITECTURE_OPENCL) =
    ( NS(arch_id_t) )SIXTRL_ARCHITECTURE_OPENCL;

SIXTRL_STATIC_VAR NS(arch_id_t) const NS(ARCHITECTURE_CUDA) =
    ( NS(arch_id_t) )SIXTRL_ARCHITECTURE_CUDA;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC_VAR NS(arch_status_t) const NS(ARCH_STATUS_SUCCESS) =
    ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_SUCCESS;

SIXTRL_STATIC_VAR NS(arch_status_t) const NS(ARCH_STATUS_GENERAL_FAILURE) =
    ( NS(arch_status_t) )SIXTRL_ARCH_STATUS_GENERAL_FAILURE;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC_VAR NS(arch_debugging_t) const
    NS(ARCH_DEBUGGING_REGISTER_EMPTY) =
        ( NS(arch_debugging_t) )SIXTRL_ARCH_DEBUGGING_REGISTER_EMPTY;

SIXTRL_STATIC_VAR NS(arch_debugging_t) const
    NS(ARCH_DEBUGGING_MIN_FLAG) =
        ( NS(arch_debugging_t) )SIXTRL_ARCH_DEBUGGING_MIN_FLAG;

SIXTRL_STATIC_VAR NS(arch_debugging_t) const
    NS(ARCH_DEBUGGING_MAX_FLAG) =
        ( NS(arch_debugging_t) )SIXTRL_ARCH_DEBUGGING_MAX_FLAG;

SIXTRL_STATIC_VAR NS(arch_debugging_t) const
    NS(ARCH_DEBUGGING_STATUS_BITMASK) =
        ( NS(arch_debugging_t) )SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK;

SIXTRL_STATIC_VAR NS(arch_debugging_t) const
    NS(ARCH_DEBUGGING_STATUS_MAX_FLAG) = ( NS(arch_debugging_t)
        )SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG;

SIXTRL_STATIC_VAR NS(arch_debugging_t) const
    NS(ARCH_DEBUGGING_GENERAL_FAILURE) =
        ( NS(arch_debugging_t) )SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC_VAR NS(arch_kernel_id_t) const
    NS(ARCH_ILLEGAL_KERNEL_ID) = ( NS(arch_kernel_id_t) )0xFFFFFFFF;

SIXTRL_STATIC_VAR NS(arch_program_id_t) const
    NS(ARCH_ILLEGAL_PROGRAM_ID) = ( NS(arch_program_id_t) )0xFFFFFFFF;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

SIXTRL_STATIC_VAR NS(node_platform_id_t) const NS(NODE_ILLEGAL_PATFORM_ID) =
    ( NS(node_platform_id_t) )-1;

SIXTRL_STATIC_VAR NS(node_device_id_t) const NS(NODE_ILLEGAL_DEVICE_ID) =
    ( NS(node_device_id_t) )-1;

SIXTRL_STATIC_VAR NS(node_index_t) const NS(NODE_UNDEFINED_INDEX) =
    ( NS(node_index_t) )0xFFFFFFFF;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_ILLEGAL_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_ILLEGAL_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_PARTICLES_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_PARTICLES_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_BEAM_ELEMENTS_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_BEAM_ELEMENTS_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_OUTPUT_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_OUTPUT_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_PARTICLE_ADDR_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_PARTICLE_ADDR_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_MIN_USER_DEFINED_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_MIN_USER_DEFINED_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_EXTERNAL_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_EXTERNAL_BUFFER_ID;

SIXTRL_STATIC_VAR NS(arch_size_t) const NS(ARCH_MAX_USER_DEFINED_BUFFER_ID) =
    ( NS(arch_size_t) )SIXTRL_ARCH_MAX_USER_DEFINED_BUFFER_ID;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

SIXTRL_STATIC_VAR NS(node_id_str_fmt_t) const NS(NODE_ID_STR_FORMAT_NOARCH) =
    ( NS(node_id_str_fmt_t) )SIXTRL_NODE_ID_STR_FORMAT_NOARCH;

SIXTRL_STATIC_VAR NS(node_id_str_fmt_t) const NS(NODE_ID_STR_FORMAT_ARCHID) =
    ( NS(node_id_str_fmt_t) )SIXTRL_NODE_ID_STR_FORMAT_ARCHID;

SIXTRL_STATIC_VAR NS(node_id_str_fmt_t) const NS(NODE_ID_STR_FORMAT_ARCHSTR) =
    ( NS(node_id_str_fmt_t) )SIXTRL_NODE_ID_STR_FORMAT_ARCHSTR;

SIXTRL_STATIC_VAR NS(node_id_str_fmt_t) const NS(NODE_ID_STR_FORMAT_DEFAULT) =
    ( NS(node_id_str_fmt_t) )SIXTRL_NODE_ID_STR_FORMAT_DEFAULT;


SIXTRL_STATIC_VAR NS(node_id_str_fmt_t) const NS(NODE_ID_STR_FORMAT_ILLEGAL) =
    ( NS(node_id_str_fmt_t) )SIXTRL_NODE_ID_STR_FORMAT_ILLEGAL;

/* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- - */

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++. Host */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    typedef ::NS(ctrl_status_t)       ctrl_status_t;
    typedef ::NS(ctrl_size_t)         ctrl_size_t;
    typedef ::NS(ctrl_kernel_id_t)    ctrl_kernel_id_t;
    typedef ::NS(ctrl_program_id_t)   ctrl_program_id_t;

    typedef ::NS(arch_status_t)       arch_status_t;
    typedef ::NS(arch_id_t)           arch_id_t;
    typedef ::NS(arch_size_t)         arch_size_t;
    typedef ::NS(arch_debugging_t)    arch_debugging_t;
    typedef ::NS(arch_kernel_id_t)    arch_kernel_id_t;
    typedef ::NS(arch_program_id_t)   arch_program_id_t;

    typedef ::NS(node_platform_id_t)  node_platform_id_t;
    typedef ::NS(node_device_id_t)    node_device_id_t;
    typedef ::NS(node_index_t)        node_index_t;
    typedef ::NS(node_id_str_fmt_t)   node_id_str_fmt_t;

    typedef enum
    {
        CTRL_PERFORM_REMAP = ::NS(CTRL_PERFORM_REMAP),
        CTRL_PERFORM_NO_REMAP = ::NS(CTRL_PERFORM_NO_REMAP)
    }
    ctrl_perform_remap_flag_t;

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_ID_BITMASK = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_ID_BITMASK );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_ID_OFFSET = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_ID_OFFSET );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_ID_VARIANT_BITMASK = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_ID_VARIANT_BITMASK );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_ID_VARIANT_OFFSET = static_cast< arch_id_t
            >( SIXTRL_ARCHITECTURE_ID_VARIANT_OFFSET );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_ILLEGAL = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_ILLEGAL );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_NONE = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_NONE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_CPU = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_CPU );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_OPENCL = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_OPENCL );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_id_t
        ARCHITECTURE_CUDA = static_cast< arch_id_t >(
            SIXTRL_ARCHITECTURE_CUDA );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_debugging_t
        ARCH_DEBUGGING_REGISTER_EMPTY = static_cast< arch_debugging_t >( 0 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_debugging_t
        ARCH_DEBUGGING_MIN_FLAG = static_cast< arch_debugging_t >(
            SIXTRL_ARCH_DEBUGGING_MIN_FLAG );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_debugging_t
        ARCH_DEBUGGING_MAX_FLAG = static_cast< arch_debugging_t >(
            SIXTRL_ARCH_DEBUGGING_MAX_FLAG );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_debugging_t
        ARCH_DEBUGGING_STATUS_BITMASK = static_cast< arch_debugging_t >(
            SIXTRL_ARCH_DEBUGGING_STATUS_BITMASK );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_debugging_t
        ARCH_DEBUGGING_STATUS_MAX_FLAG = static_cast< arch_debugging_t >(
            SIXTRL_ARCH_DEBUGGING_STATUS_MAX_FLAG );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_debugging_t
        ARCH_DEBUGGING_GENERAL_FAILURE = static_cast< arch_debugging_t >(
            SIXTRL_ARCH_DEBUGGING_GENERAL_FAILURE );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST ctrl_status_t
        ARCH_STATUS_SUCCESS = static_cast< ctrl_status_t >(
            SIXTRL_ARCH_STATUS_SUCCESS );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST ctrl_status_t
        ARCH_STATUS_GENERAL_FAILURE = static_cast< ctrl_status_t >(
            SIXTRL_ARCH_STATUS_GENERAL_FAILURE );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_kernel_id_t
        ARCH_ILLEGAL_KERNEL_ID = static_cast< arch_kernel_id_t >( 0xFFFFFFFF );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_program_id_t
        ARCH_ILLEGAL_PROGRAM_ID = static_cast< arch_program_id_t >( 0xFFFFFFFF );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_platform_id_t
        NODE_ILLEGAL_PATFORM_ID = static_cast< node_platform_id_t >( -1 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_device_id_t
        NODE_ILLEGAL_DEVICE_ID = static_cast< node_device_id_t >( -1 );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_index_t
        NODE_UNDEFINED_INDEX = static_cast< node_index_t >( 0xFFFFFFFF );

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  */

    SIXTRL_STATIC_VAR NS(arch_size_t) const ARCH_ILLEGAL_BUFFER_ID =
        ::NS(ARCH_ILLEGAL_BUFFER_ID);

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_PARTICLES_BUFFER_ID = static_cast< arch_size_t >(
            SIXTRL_ARCH_PARTICLES_BUFFER_ID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_BEAM_ELEMENTS_BUFFER_ID = ::NS(ARCH_BEAM_ELEMENTS_BUFFER_ID);

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_OUTPUT_BUFFER_ID = static_cast< arch_size_t >(
            SIXTRL_ARCH_OUTPUT_BUFFER_ID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID = static_cast< arch_size_t >(
            SIXTRL_ARCH_ELEM_BY_ELEM_CONFIG_BUFFER_ID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_PARTICLE_ADDR_BUFFER_ID = static_cast< arch_size_t >(
            SIXTRL_ARCH_PARTICLE_ADDR_BUFFER_ID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_MIN_USER_DEFINED_BUFFER_ID = static_cast< arch_size_t >(
            SIXTRL_ARCH_MIN_USER_DEFINED_BUFFER_ID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_EXTERNAL_BUFFER_ID = static_cast< arch_size_t >(
            SIXTRL_ARCH_EXTERNAL_BUFFER_ID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST arch_size_t
        ARCH_MAX_USER_DEFINED_BUFFER_ID = static_cast< arch_size_t >(
            SIXTRL_ARCH_MAX_USER_DEFINED_BUFFER_ID );

    /* -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_id_str_fmt_t
        NODE_ID_STR_FORMAT_NOARCH = static_cast< node_id_str_fmt_t >(
            SIXTRL_NODE_ID_STR_FORMAT_NOARCH );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_id_str_fmt_t
        NODE_ID_STR_FORMAT_ARCHID = static_cast< node_id_str_fmt_t >(
            SIXTRL_NODE_ID_STR_FORMAT_ARCHID );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_id_str_fmt_t
        NODE_ID_STR_FORMAT_ARCHSTR = static_cast< node_id_str_fmt_t >(
            SIXTRL_NODE_ID_STR_FORMAT_ARCHSTR );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_id_str_fmt_t
        NODE_ID_STR_FORMAT_ILLEGAL = static_cast< node_id_str_fmt_t >(
            SIXTRL_NODE_ID_STR_FORMAT_ILLEGAL );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST node_id_str_fmt_t
        NODE_ID_STR_FORMAT_DEFAULT = static_cast< node_id_str_fmt_t >(
            SIXTRL_NODE_ID_STR_FORMAT_DEFAULT );
}
#endif /* C++ */

#endif /* SIXTRACKLIB_COMMON_CONTROL_DEFINITIONS_H__ */

/* end: sixtracklib/common/control/definitions.h */
