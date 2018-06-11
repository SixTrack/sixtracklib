#ifndef SIXTRACKLIB_SIMD_TRACK_H__
#define SIXTRACKLIB_SIMD_TRACK_H__

#if !defined( _GPUCODE )

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/particles.h"
#include "sixtracklib/common/beam_elements.h"

    #if defined( __cplusplus )
    extern "C" {
    #endif /* __cplusplus */

#endif /* !defined( _GPUCODE ) */

int NS(Track_beam_elements_simd_sse2)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer );

int NS(Track_beam_elements_simd_avx)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer );


int NS(Track_drift_simd_sse2)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length );

int NS(Track_drift_simd_avx)(
    NS(Particles)* SIXTRL_RESTRICT particles, 
    NS(block_num_elements_t) const start_particle_index,
    NS(block_num_elements_t) const end_particle_index,
    SIXTRL_REAL_T const length );

#if !defined( _GPUCODE )

    #if defined( __cplusplus )
    }
    #endif /* defined( __cplusplus ) */

#endif /* !defined( GPUCODE ) */


#endif /* SIXTRACKLIB_SIMD_TRACK_H__ */

/* end: sixtracklib/simd/track.h */
