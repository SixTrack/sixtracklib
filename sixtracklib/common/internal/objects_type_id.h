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
    NS(OBJECT_TYPE_NONE)                      =          0,
    NS(OBJECT_TYPE_PARTICLE)                  =          1,
    NS(OBJECT_TYPE_DRIFT)                     =          2,
    NS(OBJECT_TYPE_DRIFT_EXACT)               =          3,
    NS(OBJECT_TYPE_MULTIPOLE)                 =          4,
    NS(OBJECT_TYPE_RF_MULTIPOLE)              =        256,
    NS(OBJECT_TYPE_CAVITY)                    =          5,
    NS(OBJECT_TYPE_XYSHIFT)                   =          6,
    NS(OBJECT_TYPE_SROTATION)                 =          7,
    NS(OBJECT_TYPE_LIN_DECOUPLED_ROTATION)    =        259,
    NS(OBJECT_TYPE_BEAM_BEAM_4D)              =          8,
    NS(OBJECT_TYPE_BEAM_BEAM_6D)              =          9,
    NS(OBJECT_TYPE_BEAM_MONITOR)              =         10,
    NS(OBJECT_TYPE_LIMIT_RECT)                =         11,
    NS(OBJECT_TYPE_LIMIT_ELLIPSE)             =         12,
    NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE)        =         16,
    NS(OBJECT_TYPE_SC_COASTING)               =         13,
    NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF)         =         14,
    NS(OBJECT_TYPE_SC_INTERPOLATED_PROF)      =        257,
    NS(OBJECT_TYPE_LINE_DENSITY_PROF_DATA)    =        258,
    NS(OBJECT_TYPE_DIPEDGE)                   =         24,
    NS(OBJECT_TYPE_PARTICLES_ADDR)            =        512,
    NS(OBJECT_TYPE_LINE)                      =       1024,
    NS(OBJECT_TYPE_ELEM_BY_ELEM_CONF)         =      65534,
    NS(OBJECT_TYPE_NODE_ID)                   =      65535,
    NS(OBJECT_TYPE_LAST_AVAILABLE)            =      65535,
    NS(OBJECT_TYPE_INVALID)                   = 0x7fffffff
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
        OBJECT_TYPE_RF_MULTIPOLE = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_RF_MULTIPOLE) );

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
        OBJECT_TYPE_BEAM_MONITOR = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BEAM_MONITOR) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LIMIT_RECT = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LIMIT_RECT) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LIMIT_ELLIPSE = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LIMIT_ELLIPSE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_LIMIT_RECT_ELLIPSE = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_LIMIT_RECT_ELLIPSE) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_SC_COASTING = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_SC_COASTING) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_SC_QGAUSSIAN_PROF = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_SC_QGAUSSIAN_PROF) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_SC_INTERPOLATED_PROF = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_SC_INTERPOLATED_PROF) );

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
        OBJECT_TYPE_BINARY_PATCH_ITEM = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_BINARY_PATCH_ITEM) );

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
        OBJECT_TYPE_MIN_USERDEFINED = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_MIN_USERDEFINED) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_TRICUB = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_TRICUB) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST object_type_id_t
        OBJECT_TYPE_TRICUB_DATA = static_cast< object_type_id_t >(
            NS(OBJECT_TYPE_TRICUB_DATA) );

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
