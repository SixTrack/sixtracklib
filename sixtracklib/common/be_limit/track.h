#ifndef SIXTRL_COMMON_BE_LIMIT_TRACK_C99_H__
#define SIXTRL_COMMON_BE_LIMIT_TRACK_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, host */

struct NS(Limit);

SIXTRL_STATIC SIXTRL_FN NS(track_status_t) NS(Track_particle_limit)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Limit) *const SIXTRL_RESTRICT limit );

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_limit/be_limit.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, host */

SIXTRL_STATIC SIXTRL_FN NS(track_status_t) NS(Track_particle_limit)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const NS(Limit) *const SIXTRL_RESTRICT limit )
{
    typedef NS(particle_real_t)  real_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0;

    real_t const x = NS(Particles_get_x_value)( particles, particle_index );
    real_t const sign_x = ( real_t )( ZERO < x ) - ( real_t )( x < ZERO  );

    real_t const y = NS(Particles_get_y_value)( particles, particle_index );
    real_t const sign_y = ( real_t )( ZERO < y ) - ( real_t )( y < ZERO  );

    index_t const new_state = ( ( index_t )(
        ( sign_x * x ) < NS(Limit_get_x_limit)( limit ) &
        ( sign_y * y ) < NS(Limit_get_y_limit)( limit ) );

    NS(Particles_set_state_value)( particles, particle_index, new_state );

    return NS(TRACK_SUCCESS);
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, host */

#endif /* SIXTRL_COMMON_BE_LIMIT_TRACK_C99_H__ */
/* end: sixtracklib/common/be_limit/track.h */
