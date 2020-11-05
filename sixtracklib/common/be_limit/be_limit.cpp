#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_limit/be_limit_rect.hpp"
    #include "sixtracklib/common/be_limit/be_limit_ellipse.hpp"
#endif/* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    constexpr TLimitRect< ::NS(particle_real_t) >::value_type
        TLimitRect< ::NS(particle_real_t) >::DEFAULT_MIN_X;

    constexpr TLimitRect< ::NS(particle_real_t) >::value_type
        TLimitRect< ::NS(particle_real_t) >::DEFAULT_MAX_X;

    constexpr TLimitRect< ::NS(particle_real_t) >::value_type
        TLimitRect< ::NS(particle_real_t) >::DEFAULT_MIN_Y;

    constexpr TLimitRect< ::NS(particle_real_t) >::value_type
        TLimitRect< ::NS(particle_real_t) >::DEFAULT_MAX_Y;

    /* --------------------------------------------------------------------- */

    constexpr TLimitEllipse< ::NS(particle_real_t) >::value_type
        TLimitEllipse< ::NS(particle_real_t) >::DEFAULT_X_HALF_AXIS;

    constexpr TLimitEllipse< ::NS(particle_real_t) >::value_type
        TLimitEllipse< ::NS(particle_real_t) >::DEFAULT_Y_HALF_AXIS;

    /* --------------------------------------------------------------------- */


}
#endif /* C++ */
