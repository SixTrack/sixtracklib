#ifndef SIXTRACKL_COMMON_BUFFER_BUFFER_TYPE_H__
#define SIXTRACKL_COMMON_BUFFER_BUFFER_TYPE_H__

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
    #include "sixtracklib/common/definitions.h"
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

#if !defined( SIXTRL_BUFFER_FLAGS_NONE )
    #define   SIXTRL_BUFFER_FLAGS_NONE                      0x00000000
#endif /* !defined( SIXTRL_BUFFER_FLAGS_NONE ) */

#if !defined( SIXTRL_BUFFER_USES_DATASTORE )
    #define   SIXTRL_BUFFER_USES_DATASTORE                  0x00000001
#endif /* !defined( SIXTRL_BUFFER_USES_DATASTORE ) */

#if !defined( SIXTRL_BUFFER_OWNS_DATASTORE )
    #define   SIXTRL_BUFFER_OWNS_DATASTORE                  0x00000002
#endif /* !defined( SIXTRL_BUFFER_OWNS_DATASTORE ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS )
    #define   SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS         0x00000004
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_CLEAR )
    #define   SIXTRL_BUFFER_DATASTORE_ALLOW_CLEAR           0x00000008
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_CLEAR ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_DELETES )
    #define   SIXTRL_BUFFER_DATASTORE_ALLOW_DELETES         0x00000010
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_DELETES ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING )
    #define   SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING       0x00000020
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_RESIZE )
    #define   SIXTRL_BUFFER_DATASTORE_ALLOW_RESIZE          0x00000040
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_ALLOW_RESIZE ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_MEMPOOL )
    #define   SIXTRL_BUFFER_DATASTORE_MEMPOOL               0x00010000
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_MEMPOOL ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_OPENCL )
    #define   SIXTRL_BUFFER_DATASTORE_OPENCL                0x00020000
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_OPENCL ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_CUDA )
    #define   SIXTRL_BUFFER_DATASTORE_CUDA                  0x00040000
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_CUDA ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_MASK )
    #define   SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_MASK    0xff000000
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_MASK ) */

#if !defined( SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_BITS )
    #define   SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_BITS    24u
#endif /* !defined( SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_BITS ) */

#if !defined( SIXTRL_BUFFER_DEFAULT_SLOT_SIZE )
    #define   SIXTRL_BUFFER_DEFAULT_SLOT_SIZE               8u
#endif /* !defined( SIXTRL_BUFFER_DEFAULT_SLOT_SIZE ) */

#if !defined( SIXTRL_BUFFER_DEFAULT_HEADER_SIZE )
    #define   SIXTRL_BUFFER_DEFAULT_HEADER_SIZE             64u
#endif /* !defined( SIXTRL_BUFFER_DEFAULT_HEADER_SIZE ) */

#if !defined( SIXTRL_BUFFER_MINIMAL_LENGTH)
    #define SIXTRL_BUFFER_MINIMAL_LENGTH                   128u
#endif /* !defined( SIXTRL_BUFFER_MINIMAL_LENGTH) */

#if !defined( _GPUCODE )

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_FLAGS_NONE) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_FLAGS_NONE;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_USES_DATASTORE) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_USES_DATASTORE;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_OWNS_DATASTORE) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_OWNS_DATASTORE;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_ALLOW_APPENDS) =
        ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_ALLOW_CLEAR) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_ALLOW_CLEAR;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_ALLOW_DELETES) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_ALLOW_DELETES;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_ALLOW_REMAPPING) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_ALLOW_RESIZE) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_ALLOW_RESIZE;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_MEMPOOL)
    = ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_MEMPOOL;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_OPENCL) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_OPENCL;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_CUDA) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_CUDA;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_SPECIAL_FLAGS_MASK) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_MASK;

SIXTRL_STATIC_VAR NS(buffer_flags_t) const NS(BUFFER_DATASTORE_SPECIAL_FLAGS_BITS) =
    ( NS(buffer_flags_t) )SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_BITS;

/* ------------------------------------------------------------------------- */

SIXTRL_STATIC_VAR NS(buffer_size_t)  const NS(BUFFER_DEFAULT_SLOT_SIZE) =
    ( NS(buffer_size_t) )SIXTRL_BUFFER_DEFAULT_SLOT_SIZE;

SIXTRL_STATIC_VAR NS(buffer_size_t) const NS(BUFFER_DEFAULT_HEADER_SIZE) =
    ( NS(buffer_size_t) )SIXTRL_BUFFER_DEFAULT_HEADER_SIZE;

SIXTRL_STATIC_VAR NS(buffer_size_t) const NS(BUFFER_MINIMAL_LENGTH) =
    ( NS(buffer_size_t) )SIXTRL_BUFFER_MINIMAL_LENGTH;

/* ------------------------------------------------------------------------- */

#endif /* !defined( _GPUCODE ) */

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */


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

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_NONE = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_NONE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_PARTICLE     = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_PARTICLE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_DRIFT        = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_DRIFT) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_DRIFT_EXACT  = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_DRIFT_EXACT) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_MULTIPOLE    = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_MULTIPOLE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_CAVITY       = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_CAVITY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_XYSHIFT      = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_XYSHIFT) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_SROTATION    = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_SROTATION) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_BEAM_BEAM_4D = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BEAM_BEAM_4D) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_BEAM_BEAM_6D = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BEAM_BEAM_6D) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LINE         = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LINE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_INVALID      = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_INVALID) );

    /* --------------------------------------------------------------------- */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_FLAGS_NONE = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_FLAGS_NONE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_USES_DATASTORE = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_USES_DATASTORE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_OWNS_DATASTORE = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_OWNS_DATASTORE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_APPENDS = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_ALLOW_APPENDS );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_CLEAR = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_ALLOW_CLEAR );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_DELETES = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_ALLOW_DELETES );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_REMAPPING = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_ALLOW_REMAPPING );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_ALLOW_RESIZE = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_ALLOW_RESIZE );

    /* --------------------------------------------------------------------- */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_MEMPOOL = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_MEMPOOL );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_OPENCL = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_OPENCL );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_CUDA = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_CUDA );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_SPECIAL_FLAGS_MASK = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_MASK );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_flags_t
        BUFFER_DATASTORE_SPECIAL_FLAGS_BITS = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DATASTORE_SPECIAL_FLAGS_BITS );

    /* --------------------------------------------------------------------- */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_size_t
        BUFFER_DEFAULT_SLOT_SIZE = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DEFAULT_SLOT_SIZE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_size_t
        BUFFER_DEFAULT_HEADER_SIZE = static_cast< buffer_flags_t >(
            SIXTRL_BUFFER_DEFAULT_HEADER_SIZE );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST buffer_size_t
        BUFFER_MINIMAL_LENGTH = static_cast< buffer_size_t >(
            SIXTRL_BUFFER_MINIMAL_LENGTH );
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_BUFFER_BUFFER_TYPE_H__ */

/* end: sixtracklib/common/buffer/buffer_type.h */
