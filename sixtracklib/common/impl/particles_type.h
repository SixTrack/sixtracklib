#ifndef SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__
#define SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/blocks.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

typedef struct NS(Particles)
{
    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        q0     __attribute__(( aligned( 8 ) ));     /* C */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        mass0  __attribute__(( aligned( 8 ) ));  /* eV */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        beta0  __attribute__(( aligned( 8 ) ));  /* nounit */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        gamma0 __attribute__(( aligned( 8 ) )); /* nounit */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        p0c    __attribute__(( aligned( 8 ) ));    /* eV */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        s      __attribute__(( aligned( 8 ) ));     /* [m] */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        x      __attribute__(( aligned( 8 ) ));     /* [m] */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        y      __attribute__(( aligned( 8 ) ));     /* [m] */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        px     __attribute__(( aligned( 8 ) ));    /* Px/P0 */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        py     __attribute__(( aligned( 8 ) ));    /* Py/P0 */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        sigma  __attribute__(( aligned( 8 ) ));
            /* s-beta0*c*t  where t is the time
                      since the beginning of the simulation */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        psigma __attribute__(( aligned( 8 ) )); /* (E-E0) / (beta0 P0c)
            conjugate of sigma */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        delta  __attribute__(( aligned( 8 ) ));  /* P/P0-1 = 1/rpp-1 */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        rpp    __attribute__(( aligned( 8 ) ));    /* ratio P0 /P */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        rvv    __attribute__(( aligned( 8 ) ));    /* ratio beta / beta0 */

    SIXTRL_GLOBAL_DEC SIXTRL_REAL_T* SIXTRL_RESTRICT
        chi    __attribute__(( aligned( 8 ) ));    /* q/q0 * m/m0  */

    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT
        particle_id __attribute__(( aligned( 8 ) ));

    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT
        lost_at_element_id __attribute__(( aligned( 8 ) )); /* element at
            which the particle was lost */

    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT
        lost_at_turn __attribute__(( aligned( 8 ) )); /* turn at which the
            particle was lost */

    SIXTRL_GLOBAL_DEC SIXTRL_INT64_T* SIXTRL_RESTRICT
        state __attribute__(( aligned( 8 ) )); /* negative means particle */

    NS(block_num_elements_t) num_of_particles  __attribute__(( aligned( 8 ) ));
}
NS(Particles);

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined( _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_PARTICLES_TYPE_H__ */

/* end: sixtracklib/common/impl/particles_type.h */

