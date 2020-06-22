#ifndef SIXTRACKLIB_COMMON_PARTICLES_DEFINITIONS_H__
#define SIXTRACKLIB_COMMON_PARTICLES_DEFINITIONS_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/particles_defines.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
extern "C" {
#endif /* C++, Host  */

/* ========================================================================= */

typedef SIXTRL_INT64_T  NS(particle_index_t);
typedef SIXTRL_REAL_T   NS(particle_real_t);
typedef SIXTRL_INT64_T  NS(particle_num_elements_t);

typedef SIXTRL_PARTICLE_DATAPTR_DEC NS(particle_real_t)*
        NS(particle_real_ptr_t);

typedef SIXTRL_PARTICLE_DATAPTR_DEC NS(particle_real_t) const*
        NS(particle_real_const_ptr_t);

typedef SIXTRL_PARTICLE_DATAPTR_DEC NS(particle_index_t)*
        NS(particle_index_ptr_t);

typedef SIXTRL_PARTICLE_DATAPTR_DEC NS(particle_index_t) const*
        NS(particle_index_const_ptr_t);

#if !defined( _GPUCODE )

SIXTRL_STATIC_VAR NS(buffer_size_t) const
    NS(PARTICLES_NUM_INDEX_DATAPTRS) = ( NS(buffer_size_t) )4u;

SIXTRL_STATIC_VAR NS(buffer_size_t) const
    NS(PARTICLES_NUM_REAL_DATAPTRS) = ( NS(buffer_size_t) )17u;

SIXTRL_STATIC_VAR NS(buffer_size_t) const
    NS(PARTICLES_NUM_DATAPTRS) = ( NS(buffer_size_t) )21u;

#else /* defined( _GPUCODE ) */

typedef enum
{
    NS(PARTICLES_NUM_INDEX_DATAPTRS) = 4,
    NS(PARTICLES_NUM_REAL_DATAPTRS)  = 17,
    NS(PARTICLES_NUM_DATAPTRS)       = 21
}
NS(ParticlesGlobalConstantsEnum);

#endif /* !defined( _GPUCODE ) */


#if defined( __cplusplus ) && !defined( _GPUCODE ) && !defined( __CUDA_ARCH__ )
}

namespace SIXTRL_CXX_NAMESPACE
{
    typedef ::NS(particle_index_t)              particle_index_t;
    typedef ::NS(particle_real_t)               particle_real_t;
    typedef ::NS(particle_num_elements_t)       particle_num_elements_t;

    typedef ::NS(particle_real_ptr_t)           particle_real_ptr_t;
    typedef ::NS(particle_real_const_ptr_t)     particle_real_const_ptr_t;

    typedef ::NS(particle_index_ptr_t)          particle_index_ptr_t;
    typedef ::NS(particle_index_const_ptr_t)    particle_index_const_ptr_t;

    /* --------------------------------------------------------------------- */

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST ::NS(buffer_size_t)
        PARTICLES_NUM_INDEX_DATAPTRS = static_cast< ::NS(buffer_size_t) >(
            ::NS(PARTICLES_NUM_INDEX_DATAPTRS) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST ::NS(buffer_size_t)
        PARTICLES_NUM_REAL_DATAPTRS = static_cast< ::NS(buffer_size_t) >(
            ::NS(PARTICLES_NUM_REAL_DATAPTRS) );

    SIXTRL_STATIC_VAR SIXTRL_CONSTEXPR_OR_CONST ::NS(buffer_size_t)
        PARTICLES_NUM_DATAPTRS = static_cast< ::NS(buffer_size_t) >(
            ::NS(PARTICLES_NUM_DATAPTRS) );
}

#endif /* C++, Host  */
#endif /* SIXTRACKLIB_COMMON_PARTICLES_DEFINITIONS_H__ */
/* end: sixtracklib/common/particles/definitions.h */
