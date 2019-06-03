#ifndef SIXTRL_COMMON_BE_DIPEDGE_TRACK_C99_H__
#define SIXTRL_COMMON_BE_DIPEDGE_TRACK_C99_H__

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

struct NS(DipoleEdge);

SIXTRL_STATIC SIXTRL_FN NS(track_status_t) NS(Track_particle_dipedge)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BE_ARGPTR_DEC const struct NS(DipoleEdge) 
        *const SIXTRL_RESTRICT dipedge);

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, host */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_dipedge/be_dipedge.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE )
extern "C" {
#endif /* C++, host */

SIXTRL_INLINE NS(track_status_t) NS(Track_particle_dipedge)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BE_ARGPTR_DEC const NS(DipoleEdge) *const SIXTRL_RESTRICT dipedge )
{
    NS(Particles_add_to_px_value)( particles, particle_idx, 
       NS(DipoleEdge_get_inv_rho)( dipedge ) * 
       NS(DipoleEdge_get_tan_rot_angle)( dipedge ) * 
       NS(Particles_get_x_value)( particles, particle_idx ) );
    
    return NS(TRACK_SUCCESS);
}

#if defined( __cplusplus ) && !defined( _GPUCODE )
}
#endif /* C++, host */

#endif /* SIXTRL_COMMON_BE_DIPEDGE_TRACK_C99_H__ */
/* end: sixtracklib/common/be_dipedge/track.h */
