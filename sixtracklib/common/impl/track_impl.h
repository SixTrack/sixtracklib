#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_IMPL_TACK_IMPL_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_IMPL_TACK_IMPL_H__

/* ************************************************************************** */

#ifndef SIXTRACKLIB_DRIFT_TRACK_IMPL
#define SIXTRACKLIB_DRIFT_TRACK_IMPL( T, particles, ip, length )  \
        \
    /* ------------------------------------------------------------------------- */ \
    /* -----   Inside SIXTRACKLIB_DRIFT_TRACK_IMPL                               */ \
    /* ------------------------------------------------------------------------- */ \
        \
    T const ONE    = ( T )1; \
    T const TWO    = ( T )2; \
    T const rpp    = NS(Particles_get_rpp)(   ( particles ), ( ip ) ); \
    T const rvv    = NS(Particles_get_rvv)(   ( particles ), ( ip ) ); \
    T const px     = NS(Particles_get_px)(    ( particles ), ( ip ) ) * rpp; \
    T const py     = NS(Particles_get_py)(    ( particles ), ( ip ) ) * rpp; \
    T const dsigma = ( ONE - rvv * ( ONE + ( px * px + py * py ) / TWO ) );  \
        \
    T x            = NS(Particles_get_x)(     ( particles ), ( ip ) ); \
    T y            = NS(Particles_get_y)(     ( particles ), ( ip ) ); \
    T s            = NS(Particles_get_s)(     ( particles ), ( ip ) ); \
    T sigma        = NS(Particles_get_sigma)( ( particles ), ( ip ) ); \
        \
    x             += ( length ) * px; \
    y             += ( length ) * py; \
    sigma         += ( length ) * dsigma; \
    s             += ( length ); \
        \
    NS(Particles_set_sigma)( ( particles ), ( ip ), sigma ); \
    NS(Particles_set_x)(     ( particles ), ( ip ), x     ); \
    NS(Particles_set_y)(     ( particles ), ( ip ), y     ); \
    NS(Particles_set_s)(     ( particles ), ( ip ), s     ); \
        \
    /* ------------------------------------------------------------------------- */ \
    /* -----   End Of SIXTRACKLIB_DRIFT_TRACK_IMPL                               */ \
    /* ------------------------------------------------------------------------- */ \

#endif /* !defined( SIXTRACKLIB_DRIFT_TRACK_IMPL ) */

/* ************************************************************************** */

#ifndef SIXTRACKLIB_DRIFT_EXACT_TRACK_IMPL
#define SIXTRACKLIB_DRIFT_EXACT_TRACK_IMPL( T, particles, ip, length ) \
    \
    /* ------------------------------------------------------------------------- */ \
    /* -----   Inside SIXTRACKLIB_DRIFT_EXACT_TRACK_IMPL                         */ \
    /* ------------------------------------------------------------------------- */ \
        \
    T const ONE   = ( T )1; \
    T const delta = NS(Particles_get_delta)( ( particles ), ( ip ) ); \
    T const beta0 = NS(Particles_get_beta0)( ( particles ), ( ip ) ); \
    T sigma       = NS(Particles_get_sigma)( ( particles ), ( ip ) ); \
    T const px    = NS(Particles_get_px)(    ( particles ), ( ip ) ); \
    T const py    = NS(Particles_get_py)(    ( particles ), ( ip ) ); \
     \
    T const opd   = delta + ONE; \
    T const lpzi  = ( length ) / sqrt( opd * opd - px * px - py * py ); \
    T const lbzi  = ( beta0 * beta0 * sigma + ONE ) * lpzi; \
        \
    T x           = NS(Particles_get_x)( ( particles ), ( ip ) ); \
    T y           = NS(Particles_get_y)( ( particles ), ( ip ) ); \
    T s           = NS(Particles_get_s)( ( particles ), ( ip ) ); \
        \
    x            += px * lpzi; \
    y            += py * lpzi; \
    s            += ( length ); \
    sigma        += ( length ) - lbzi; \
        \
    NS(Particles_set_x)(     ( particles ), ( ip ), x     ); \
    NS(Particles_set_y)(     ( particles ), ( ip ), y     ); \
    NS(Particles_set_s)(     ( particles ), ( ip ), s     ); \
    NS(Particles_set_sigma)( ( particles ), ( ip ), sigma ); \
        \
    /* ------------------------------------------------------------------------- */ \
    /* -----   End Of SIXTRACKLIB_DRIFT_EXACT_TRACK_IMPL                         */ \
    /* ------------------------------------------------------------------------- */ \

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_IMPL_TACK_IMPL_H__ */

/* ************************************************************************** */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_IMPL_TACK_IMPL_H__ */
/* end: sixtracklib/common/impl/track_impl.h */
