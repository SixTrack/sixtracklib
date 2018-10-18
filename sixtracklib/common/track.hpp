#ifndef CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__
#define CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <cstddef>
    #include <cstdint>
    #include <cstdlib>
    #include <limits>
    #include <utility>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )

struct ::NS(Particles);
struct ::NS(Drift);
struct ::NS(DriftExact);
struct ::NS(MultiPole);
struct ::NS(Cavity);
struct ::NS(XYShift);
struct ::NS(SRotation);

namespace SIXTRL_NAMESPACE
{
    template< class ParticlesT, class DriftT >
    SIXTRL_FN bool Track_particle_drift(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t particle_index,
        SIXTRL_DATAPTR_DEC DriftT* SIXTRL_RESTRICT drift );

    SIXTRL_FN bool Track_particle_drift(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT particles,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const struct ::NS(Drift) *const
            SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class DriftExactT >
    SIXTRL_FN bool Track_particle_drift_exact(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t const particle_index,
        SIXTRL_DATAPTR_DEC DriftExactT* SIXTRL_RESTRICT drift );

    SIXTRL_FN bool Track_particle_drift_exact(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT particles,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const ::NS(DriftExact) *const
            SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class MultiPoleT >
    SIXTRL_FN bool Track_particle_multipole(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t particle_index,
        SIXTRL_DATAPTR_DEC const MultiPoleT *const SIXTRL_RESTRICT mp );

    SIXTRL_FN bool Track_particle_drift(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT particles,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const struct ::NS(MultiPole) *const
            SIXTRL_RESTRICT mp ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class XYShiftT >
    SIXTRL_FN bool Track_particle_xy_shift(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        SIXTRL_DATAPTR_DEC const XYShiftT *const SIXTRL_RESTRICT xy_shift,
        typename ParticlesT::num_elements_t particle_index );

    SIXTRL_FN bool Track_particle_xy_shift(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT particles,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const struct ::NS(XYShift) *const
            SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class SRotationT >
    SIXTRL_FN bool Track_particle_srotation(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t particle_index,
        SIXTRL_DATAPTR_DEC SRotationT* SIXTRL_RESTRICT srotation );

    SIXTRL_FN bool Track_particle_srotation(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT particles,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const struct ::NS(SRotation) *const
            SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class CavityT >
    SIXTRL_FN bool Track_particle_cavity(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t particle_index,
        SIXTRL_DATAPTR_DEC const CavityT *const SIXTRL_RESTRICT cavity );

    SIXTRL_FN bool Track_particle_cavity(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT particles,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const struct ::NS(Cavity) *const
            SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT;

    /* --------------------------------------------------------------------- */
}

/* ************************************************************************* *
 * **** Inline method implementation                                         *
 * ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/beam_elements.h"
    #include "sixtracklib/track.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */


namespace SIXTRL_NAMESPACE
{
    template< class ParticlesT, class DriftT >
    SIXTRL_INLINE bool Track_particle_drift(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT p,
        typename ParticlesT::num_elements_t index,
        SIXTRL_DATAPTR_DEC DriftT* SIXTRL_RESTRICT drift )
    {
        using particles_t = ParticlesT;
        using drift_t     = DriftT;
        using real_t      = typename particles_t::value_type;

        SIXTRL_ASSERT( p     != nullptr );
        SIXTRL_ASSERT( drift != nullptr );

        real_t const rpp    = p->getRppValue( index );
        real_t const xp     = p->getPxValue(  index ) * rpp;
        real_t const yp     = p->getPyValue(  index ) * rpp;
        real_t const rvv    = p->getRvvValue( index );

        real_t const dzeta  =
            rvv - real_t{ 1 } + real_t{ 0.5 } * ( xp*xp + yp*yp );

        real_t const length = drift->getLength();

        real_t s    = p->getSValue( index );
        real_t x    = p->getXValue( index );
        real_t y    = p->getYValue( index );
        real_t zeta = p->getZetaValue( index );

        s    += length );
        x    += length * xp );
        y    += length * yp );
        zeta += length * dzeta );

        p->setSValue( index, s );
        p->setXValue( index, x );
        p->setYValue( index, y );
        p->setZetaValue( index, zeta );

        return true;
    }

    SIXTRL_INLINE bool Track_particle_drift(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT particles,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const struct ::NS(Drift) *const
            SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
    {
        return ( 0 == ::NS(Track_particle_drift)(
            particles, particle_index, drift ) );
    }

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class DriftExactT >
    SIXTRL_INLINE bool Track_particle_drift_exact(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t const index,
        SIXTRL_DATAPTR_DEC DriftExactT* SIXTRL_RESTRICT drift );

    SIXTRL_INLINE bool Track_particle_drift_exact(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT p,
        ::NS(particle_num_elements_t) const particle_index,
        SIXTRL_DATAPTR_DEC const ::NS(DriftExact) *const
            SIXTRL_RESTRICT drift ) SIXTRL_NOEXCEPT
    {
        return ( 0 == ::NS(Track_particle_drift_exact)( p, index, drift ) );
    }

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class MultiPoleT >
    SIXTRL_INLINE bool Track_particle_multipole(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t particle_index,
        SIXTRL_DATAPTR_DEC const MultiPoleT *const SIXTRL_RESTRICT mp );

    SIXTRL_INLINE bool Track_particle_drift(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT p,
        ::NS(particle_num_elements_t) const index,
        SIXTRL_DATAPTR_DEC const struct ::NS(MultiPole) *const
            SIXTRL_RESTRICT mp ) SIXTRL_NOEXCEPT
    {
        return ( 0 == ::NS(Track_particle_multipole)( p, index, mp ) );
    }

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class XYShiftT >
    SIXTRL_INLINE bool Track_particle_xy_shift(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        SIXTRL_DATAPTR_DEC const XYShiftT *const SIXTRL_RESTRICT xy_shift,
        typename ParticlesT::num_elements_t particle_index );

    SIXTRL_INLINE bool Track_particle_xy_shift(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT p,
        ::NS(particle_num_elements_t) const index,
        SIXTRL_DATAPTR_DEC const struct ::NS(XYShift) *const
            SIXTRL_RESTRICT xy_shift ) SIXTRL_NOEXCEPT
    {
        return ( 0 == ::NS(Track_particle_xy_shift)( p, index, xy_shift ) );
    }

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class SRotationT >
    SIXTRL_INLINE bool Track_particle_srotation(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t particle_index,
        SIXTRL_DATAPTR_DEC SRotationT* SIXTRL_RESTRICT srotation );

    SIXTRL_INLINE bool Track_particle_srotation(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT p,
        ::NS(particle_num_elements_t) const index,
        SIXTRL_DATAPTR_DEC const struct ::NS(SRotation) *const
            SIXTRL_RESTRICT srotation ) SIXTRL_NOEXCEPT
    {
        return ( 0 == ::NS(Track_particle_srotation)( p, index, srotation ) );
    }

    /* --------------------------------------------------------------------- */

    template< class ParticlesT, class CavityT >
    SIXTRL_INLINE bool Track_particle_cavity(
        SIXTRL_ARGPTR_DEC ParticlesT* SIXTRL_RESTRICT particles,
        typename ParticlesT::num_elements_t particle_index,
        SIXTRL_DATAPTR_DEC const CavityT *const SIXTRL_RESTRICT cavity );

    SIXTRL_INLINE bool Track_particle_cavity(
        SIXTRL_ARGPTR_DEC struct ::NS(Particles)* SIXTRL_RESTRICT p,
        ::NS(particle_num_elements_t) const index,
        SIXTRL_DATAPTR_DEC const struct ::NS(Cavity) *const
            SIXTRL_RESTRICT cavity ) SIXTRL_NOEXCEPT
    {
        return ( 0 == ::NS(Track_particle_cavity)( p, index, cavity ) );
    }

    /* --------------------------------------------------------------------- */
}

#endif /* CXX_SIXTRACKLIB_COMMON_BUFFER_HPP__ */

/* end: sixtracklib/common/track.hpp */
