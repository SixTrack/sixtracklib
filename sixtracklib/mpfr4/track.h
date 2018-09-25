#ifndef SIXTRACKLIB_MPFR4_TRACK_H__
#define SIXTRACKLIB_MPFR4_TRACK_H__

#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#include <mpfr.h>

#include "sixtracklib/_impl/namespace_begin.h"

#if defined( __cplusplus )
extern "C" {
#endif /* defined( __cplusplus ) */

typedef struct st_MpfrWrapper
{
    mpfr_t value;
}
st_MpfrWrapper;

#if defined( SIXTRL_REAL_T )
#undef SIXTRL_REAL_T
#endif /* defined( SIXTRL_REAL_T ) */

#if !defined( SIXTRL_REAL_T )
    #define SIXTRL_REAL_T st_MpfrWrapper
#endif /* !defined( SIXTRL_REAL_T ) */

#include "sixtracklib/_impl/definitions.h"
#include "sixtracklib/common/impl/block_info_impl.h"
#include "sixtracklib/common/impl/particles_impl.h"
#include "sixtracklib/common/beam_elements.h"
#include "sixtracklib/common/particles.h"

int NS(Track_beam_elements_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    const NS(BeamElements) *const SIXTRL_RESTRICT beam_elements,
    mpfr_prec_t const prec, mpfr_rnd_t const rnd,
    NS(block_num_elements_t) const elem_by_elem_start_index,
    NS(ParticlesContainer)* SIXTRL_RESTRICT elem_by_elem_buffer );

int NS(Track_drift_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_REAL_T const drift_length,
    mpfr_prec_t const prec, mpfr_rnd_t const rnd );

int NS(Track_drift_exact_mpfr4)(
    NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_REAL_T const drift_length,
    mpfr_prec_t const prec, mpfr_rnd_t const rnd );

#if defined( __cplusplus )
}
#endif /* defined( __cplusplus ) */

#endif /* SIXTRACKLIB_MPFR4_TRACK_H__ */

/* end: sixtracklib/mpfr4/track.h */
