#ifndef SIXTRACKL_COMMON_IMPL_BUFFER_TYPE_H__
#define SIXTRACKL_COMMON_IMPL_BUFFER_TYPE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
    #else /* defined( __cplusplus ) */
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
        #include <limits.h>
    #endif /* defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */

typedef SIXTRL_UINT64_T NS(buffer_addr_t);
typedef SIXTRL_INT64_T  NS(buffer_addr_diff_t);
typedef SIXTRL_UINT64_T NS(buffer_size_t);
typedef SIXTRL_UINT64_T NS(buffer_flags_t);
typedef SIXTRL_UINT64_T NS(object_type_id_t);

typedef enum NS(object_type_values_e)
{
    NS(OBJECT_TYPE_NONE)         = 0x00000000,
    NS(OBJECT_TYPE_PARTICLE)     = 0x00000001,
    NS(OBJECT_TYPE_DRIFT)        = 0x00000002,
    NS(OBJECT_TYPE_DRIFT_EXACT)  = 0x00000003,
    NS(OBJECT_TYPE_MULTIPOLE)    = 0x00000004,
    NS(OBJECT_TYPE_CAVITY)       = 0x00000005,
    NS(OBJECT_TYPE_XYSHIFT)      = 0x00000006,
    NS(OBJECT_TYPE_SROTATION)    = 0x00000007,
    NS(OBJECT_TYPE_BEAM_BEAM_4D) = 0x00000008,
    NS(OBJECT_TYPE_BEAM_BEAM_6D) = 0x00000009,
    NS(OBJECT_TYPE_LINE)         = 0x00000100,
    NS(OBJECT_TYPE_INVALID)      = 0x7fffffff
}
NS(object_type_values_t);

#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_FLAGS_NONE)                   = ( NS(buffer_flags_t) )0x00000000;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_USES_DATASTORE)               = ( NS(buffer_flags_t) )0x00000001;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_OWNS_DATASTORE)               = ( NS(buffer_flags_t) )0x00000002;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_ALLOW_APPENDS)      = ( NS(buffer_flags_t) )0x00000004;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_ALLOW_CLEAR)        = ( NS(buffer_flags_t) )0x00000008;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_ALLOW_DELETES)      = ( NS(buffer_flags_t) )0x00000010;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_ALLOW_REMAPPING)    = ( NS(buffer_flags_t) )0x00000020;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_ALLOW_RESIZE)       = ( NS(buffer_flags_t) )0x00000040;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_MEMPOOL)            = ( NS(buffer_flags_t) )0x00010000;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_OPENCL)             = ( NS(buffer_flags_t) )0x00020000;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_CUDA)               = ( NS(buffer_flags_t) )0x00040000;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_SPECIAL_FLAGS_MASK) = ( NS(buffer_flags_t) )0xff000000;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const
    NS(BUFFER_DATASTORE_SPECIAL_FLAGS_BITS) = ( NS(buffer_flags_t) )24u;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC_VAR NS(buffer_size_t)  const
    NS(BUFFER_DEFAULT_SLOT_SIZE)            = ( NS(buffer_size_t) )8u;

SIXTRL_STATIC_VAR NS(buffer_size_t) const
    NS(BUFFER_DEFAULT_HEADER_SIZE)          = ( NS(buffer_size_t) )64u;

/* ------------------------------------------------------------------------- */

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

typedef struct NS(Buffer)
{
    NS(buffer_addr_t)    data_addr        SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    data_size        SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    header_size      SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    data_capacity    SIXTRL_ALIGN( 8u );

    NS(buffer_size_t)    slot_length      SIXTRL_ALIGN( 8u );

    NS(buffer_addr_t)    object_addr      SIXTRL_ALIGN( 8u );
    NS(buffer_size_t)    num_objects      SIXTRL_ALIGN( 8u );

    NS(buffer_flags_t)   datastore_flags  SIXTRL_ALIGN( 8u );
    NS(buffer_addr_t)    datastore_addr   SIXTRL_ALIGN( 8u );
}
NS(Buffer);

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined( __cplusplus )

namespace SIXTRL_NAMESPACE
{
    using object_type_id_t = NS(object_type_id_t);
    using buffer_addr_t    = NS(buffer_addr_t);
    using buffer_size_t    = NS(buffer_size_t);
    using buffer_flags_t   = NS(buffer_flags_t);

    /* --------------------------------------------------------------------- */

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_NONE         = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_NONE) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_PARTICLE     = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_PARTICLE) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_DRIFT        = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_DRIFT) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_DRIFT_EXACT  = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_DRIFT_EXACT) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_MULTIPOLE    = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_MULTIPOLE) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_CAVITY       = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_CAVITY) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_XYSHIFT      = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_XYSHIFT) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_SROTATION    = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_SROTATION) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_BEAM_BEAM_4D = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BEAM_BEAM_4D) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_BEAM_BEAM_6D = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BEAM_BEAM_6D) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LINE         = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LINE) );

    static SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_INVALID      = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_INVALID) );

    /* --------------------------------------------------------------------- */

    #if !defined( _GPUCODE )

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_FLAGS_NONE = NS(BUFFER_FLAGS_NONE);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_USES_DATASTORE = NS(BUFFER_USES_DATASTORE);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_OWNS_DATASTORE = NS(BUFFER_OWNS_DATASTORE);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_APPENDS = NS(BUFFER_DATASTORE_ALLOW_APPENDS);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_CLEAR = NS(BUFFER_DATASTORE_ALLOW_CLEAR);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_DELETES = NS(BUFFER_DATASTORE_ALLOW_DELETES);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_REMAPPING =
            NS(BUFFER_DATASTORE_ALLOW_REMAPPING);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_RESIZE =
            NS(BUFFER_DATASTORE_ALLOW_RESIZE);

    /* --------------------------------------------------------------------- */

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_MEMPOOL = NS(BUFFER_DATASTORE_MEMPOOL);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_OPENCL = NS(BUFFER_DATASTORE_OPENCL);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_CUDA = NS(BUFFER_DATASTORE_CUDA);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_SPECIAL_FLAGS_MASK =
            NS(BUFFER_DATASTORE_SPECIAL_FLAGS_MASK);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_SPECIAL_FLAGS_BITS =
            NS(BUFFER_DATASTORE_SPECIAL_FLAGS_BITS);

    /* --------------------------------------------------------------------- */

    static SIXTRL_CONSTEXPR_OR_CONST buffer_size_t
        BUFFER_DEFAULT_SLOT_SIZE   = NS(BUFFER_DEFAULT_SLOT_SIZE);

    static SIXTRL_CONSTEXPR_OR_CONST buffer_size_t
        BUFFER_DEFAULT_HEADER_SIZE = NS(BUFFER_DEFAULT_HEADER_SIZE);

    #endif /* !defined( _GPUCODE ) */
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_IMPL_BUFFER_TYPE_H__ */

/* end: sixtracklib/common/impl/buffer_type.h */
