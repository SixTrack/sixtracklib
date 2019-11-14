#ifndef SIXTRACKL_COMMON_INTERNAL_OBJECTS_TYPE_ID_H__
#define SIXTRACKL_COMMON_INTERNAL_OBJECTS_TYPE_ID_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

typedef enum NS(object_type_values_e)
{
    NS(OBJECT_TYPE_NONE)                  =          0,
    NS(OBJECT_TYPE_PARTICLE)              =          1,
    NS(OBJECT_TYPE_DRIFT)                 =          2,
    NS(OBJECT_TYPE_DRIFT_EXACT)           =          3,
    NS(OBJECT_TYPE_MULTIPOLE)             =          4,
    NS(OBJECT_TYPE_CAVITY)                =          5,
    NS(OBJECT_TYPE_XYSHIFT)               =          6,
    NS(OBJECT_TYPE_SROTATION)             =          7,
    NS(OBJECT_TYPE_BEAM_BEAM_4D)          =          8,
    NS(OBJECT_TYPE_BEAM_BEAM_6D)          =          9,
    NS(OBJECT_TYPE_BEAM_MONITOR)          =         10,
    NS(OBJECT_TYPE_LIMIT_RECT)            =         11,
    NS(OBJECT_TYPE_LIMIT_ELLIPSE)         =         12,
    NS(OBJECT_TYPE_SPACE_CHARGE_COASTING) =         13,
    NS(OBJECT_TYPE_SPACE_CHARGE_BUNCHED)  =         14,
    NS(OBJECT_TYPE_DIPEDGE)               =         24,
    NS(OBJECT_TYPE_PARTICLES_ADDR)        =        512,
    NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM)   =        513,
    NS(OBJECT_TYPE_LINE)                  =       1024,
    NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF)     =       1025,
    NS(OBJECT_TYPE_NODE_ID)               =       1026,
    NS(OBJECT_TYPE_BINARY_ARRAY)          =       8192,
    NS(OBJECT_TYPE_REAL_ARRAY)            =       8193,
    NS(OBJECT_TYPE_FLOAT32_ARRAY)         =       8194,
    NS(OBJECT_TYPE_UINT64_ARRAY)          =       8195,
    NS(OBJECT_TYPE_INT64_ARRAY)           =       8196,
    NS(OBJECT_TYPE_UINT32_ARRAY)          =       8197,
    NS(OBJECT_TYPE_INT32_ARRAY)           =       8198,
    NS(OBJECT_TYPE_UINT16_ARRAY)          =       8199,
    NS(OBJECT_TYPE_INT16_ARRAY)           =       8200,
    NS(OBJECT_TYPE_UINT8_ARRAY)           =       8192,
    NS(OBJECT_TYPE_INT8_ARRAY)            =       8201,
    NS(OBJECT_TYPE_CSTRING)               =       8202,
    NS(OBJECT_TYPE_BINARY_VECTOR)         =       8203,
    NS(OBJECT_TYPE_MIN_USERDEFINED)       =      32768,
    NS(OBJECT_TYPE_MAX_USERDEFINED)       =      65535,
    NS(OBJECT_TYPE_LAST_AVAILABLE)        =      65535,
    NS(OBJECT_TYPE_INVALID)               = 0x7fffffff
}
NS(object_type_values_t);

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#if defined( __cplusplus )

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

namespace SIXTRL_CXX_NAMESPACE
{
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
        OBJECT_TYPE_BEAM_MONITOR = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BEAM_MONITOR) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LIMIT_RECT = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LIMIT_RECT) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LIMIT_ELLIPSE = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LIMIT_ELLIPSE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_DIPEDGE = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_DIPEDGE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_PARTICLES_ADDR = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_PARTICLES_ADDR) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_ASSIGN_ADDRESS_ITEM = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_ASSIGN_ADDRESS_ITEM) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LINE         = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LINE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_NODE_ID = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_NODE_ID) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_ELEM_BY_ELEM_CONF = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_BINARY_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BINARY_ARRAY ) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_REAL_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_REAL_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_UINT64_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_UINT64_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_INT64_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_INT64_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_UINT32_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_UINT32_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_INT32_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_INT32_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_UINT16_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_UINT16_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_INT16_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_INT16_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_UINT8_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_UINT8_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_INT8_ARRAY = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_INT8_ARRAY) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_CSTRING = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_CSTRING) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_BINARY_VECTOR = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BINARY_VECTOR) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_MIN_USERDEFINED = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_MIN_USERDEFINED) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_MAX_USERDEFINED = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_MAX_USERDEFINED) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LAST_AVAILABLE = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LAST_AVAILABLE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_INVALID = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_INVALID) );

    template< class Elem >
    struct ObjectTypeTraits
    {
        SIXTRL_STATIC SIXTRL_INLINE object_type_id_t Type() SIXTRL_NOEXCEPT
        {
            return NS(OBJECT_TYPE_INVALID);
        }
    };
}

#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKL_COMMON_INTERNAL_OBJECTS_TYPE_ID_H__ */

/* end: sixtracklib/common/internal/beam_elements_type_id.h */
