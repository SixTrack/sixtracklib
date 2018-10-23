#ifndef SIXTRACKLIB_COMMON_TRACK_H__
#define SIXTRACKLIB_COMMON_TRACK_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(Drift);
struct NS(DriftExact);
struct NS(MultiPole);
struct NS(Cavity);
struct NS(XYShift);
struct NS(SRotation);
/* struct NS(BeamBeam); */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_drift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Drift) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_drift_exact)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t)  const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(DriftExact) *const SIXTRL_RESTRICT drift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t)  const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(XYShift) *const SIXTRL_RESTRICT xy_shift );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_srotation)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(SRotation) *const SIXTRL_RESTRICT srotation );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_cavity)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(Cavity) *const SIXTRL_RESTRICT cavity );


/* ========================================================================= */

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const beam_elements );

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const beam_elements,
    NS(buffer_size_t) const beam_element_index );

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_index );

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_begin_index,
    NS(buffer_size_t) const beam_element_end_index );

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements );

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_subset_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_index );

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_subset_beam_element_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_begin_index,
    NS(buffer_size_t) const beam_element_end_index );

SIXTRL_HOST_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_subset_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements );

#endif /* !defined( _GPUCODE ) */

/* ========================================================================= */

struct NS(Object);

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end );


SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_subset_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const index_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object)
        const* SIXTRL_RESTRICT be_info );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particle_subset_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const index_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object)
        const* SIXTRL_RESTRICT be_info_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object)
        const* SIXTRL_RESTRICT be_info_end );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN
NS(Track_particles_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object)
        const* SIXTRL_RESTRICT be_info_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC struct NS(Object)
        const* SIXTRL_RESTRICT be_info_end );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */
/* =====        Implementation of Inline functions and methods         ===== */
/* ========================================================================= */

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stddef.h>
    #include <stdint.h>
    #include <stdlib.h>
    #include <math.h>
#endif /* #if !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/generated/config.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/beam_elements.h"
    #include "sixtracklib/common/buffer/buffer_type.h"
    #if !defined( SIXTRL_DISABLE_BEAM_BEAM )
        #include "sixtracklib/common/be_beambeam/track_beambeam.h"
    #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_drift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(Drift) *const SIXTRL_RESTRICT drift )
{
    typedef NS(particle_real_t) real_t;

    real_t const rpp    = NS(Particles_get_rpp_value)( p, ii );
    real_t const xp     = NS(Particles_get_px_value )( p, ii ) * rpp;
    real_t const yp     = NS(Particles_get_py_value )( p, ii ) * rpp;
    real_t const length = NS(Drift_get_length)( drift );
    real_t const dzeta  = NS(Particles_get_rvv_value)( p, ii ) -
                          ( ( real_t )1 + ( xp*xp + yp*yp ) / ( real_t )2 );

    SIXTRL_ASSERT( NS(Particles_get_beta0_value)( p, ii ) > ( real_t )0 );

    NS(Particles_add_to_s_value)( p, ii, length );
    NS(Particles_add_to_x_value)( p, ii, length * xp );
    NS(Particles_add_to_y_value)( p, ii, length * yp );
    NS(Particles_add_to_zeta_value)( p, ii, length * dzeta );

    #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
        SIXTRL_ENABLE_APERATURE_CHECK == 1

    typedef NS(particle_index_t) index_t;

    printf( "enabled\r\n" );

    /* state == 1 -> particle is valid
     *       == 0 -> particle is outside of the aperature limits */

    if( NS(Particles_get_state_value)( p, ii ) == ( index_t )1u )
    {
        SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0;

        real_t const x = NS(Particles_get_x_value)( p, ii );
        real_t const y = NS(Particles_get_y_value)( p, ii );

        real_t const sign_x = ( real_t )( ( ZERO < x ) - ( real_t )( x < ZERO ) );
        real_t const sign_y = ( real_t )( ( ZERO < y ) - ( real_t )( y < ZERO ) );

        NS(Particles_set_state_value)( p, ii,
            ( index_t )( ( ( sign_x * x ) < SIXTRL_APERATURE_X_LIMIT ) &
                         ( ( sign_y * y ) < SIXTRL_APERATURE_Y_LIMIT ) ) );
    }

    #endif /* SIXTRL_ENABLE_APERATURE_CHECK */

    return ( SIXTRL_TRACK_RETURN )0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_drift_exact)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(DriftExact) *const SIXTRL_RESTRICT drift )
{
    typedef NS(particle_real_t) real_t;

    real_t const length = NS(DriftExact_get_length)( drift );
    real_t const px     = NS(Particles_get_px_value)( p, ii );
    real_t const py     = NS(Particles_get_py_value)( p, ii );

    real_t const opd    = NS(Particles_get_delta_value)( p, ii ) + ( real_t )1;
    real_t const lpzi   = length / sqrt( opd * opd - ( px * px + py * py ) );

    real_t const dzeta  = NS(Particles_get_rvv_value)( p, ii ) * length
                        - opd * lpzi;

    NS(Particles_add_to_s_value)(    p, ii, length );
    NS(Particles_add_to_x_value)(    p, ii, px * lpzi );
    NS(Particles_add_to_y_value)(    p, ii, py * lpzi );
    NS(Particles_add_to_zeta_value)( p, ii, dzeta );

    SIXTRL_ASSERT( NS(Particles_get_beta0_value)( p, ii ) > ( real_t )0 );
    SIXTRL_ASSERT( ( opd * opd ) >   ( px * px + py * py ) );
    SIXTRL_ASSERT( sqrt( opd * opd - ( px * px + py * py ) ) > ( real_t )0 );

    #if defined( SIXTRL_ENABLE_APERATURE_CHECK ) && \
        SIXTRL_ENABLE_APERATURE_CHECK == 1

    typedef NS(particle_index_t) index_t;

    /* state == 1 -> particle is valid
     *       == 0 -> particle is outside of the aperature limits */

    if( NS(Particles_get_state_value)( p, ii ) == ( index_t )1u )
    {
        SIXTRL_STATIC_VAR real_t const ZERO = ( real_t )0;

        real_t const x = NS(Particles_get_x_value)( p, ii );
        real_t const y = NS(Particles_get_y_value)( p, ii );

        real_t const sign_x = ( real_t )( ( ZERO < x ) - ( real_t )( x < ZERO ) );
        real_t const sign_y = ( real_t )( ( ZERO < y ) - ( real_t )( y < ZERO ) );

        NS(Particles_set_state_value)( p, ii,
            ( index_t )( ( ( sign_x * x ) < SIXTRL_APERATURE_X_LIMIT ) &&
                         ( ( sign_y * y ) < SIXTRL_APERATURE_Y_LIMIT ) ) );
    }

    #endif /* SIXTRL_ENABLE_APERATURE_CHECK */

    return ( SIXTRL_TRACK_RETURN )0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT mp )
{
    typedef NS(particle_real_t)  real_t;
    typedef NS(particle_index_t) index_t;

    SIXTRL_STATIC_VAR index_t const TWO  = ( index_t )2;
    SIXTRL_STATIC_VAR real_t  const ZERO = ( real_t )0.0;

    index_t const order = NS(MultiPole_get_order)( mp );
    index_t index_x = TWO * order;
    index_t index_y = index_x + ( index_t )1;

    real_t dpx = NS(MultiPole_get_bal_value)( mp, index_x );
    real_t dpy = NS(MultiPole_get_bal_value)( mp, index_y );

    real_t const x      = NS(Particles_get_x_value)( particles, ii );
    real_t const y      = NS(Particles_get_y_value)( particles, ii );
    real_t const chi    = NS(Particles_get_chi_value)( particles, ii );

    real_t const hxl    = NS(MultiPole_get_hxl)( mp );
    real_t const hyl    = NS(MultiPole_get_hyl)( mp );

    while( index_x > 0 )
    {
        real_t const zre = dpx * x - dpy * y;
        real_t const zim = dpx * y + dpy * x;

        SIXTRL_ASSERT( index_x >= TWO );
        SIXTRL_ASSERT( index_y >= TWO );

        index_x -= TWO;
        index_y -= TWO;

        dpx = NS(MultiPole_get_bal_value)( mp, index_x ) + zre;
        dpy = NS(MultiPole_get_bal_value)( mp, index_y ) + zim;
    }

    dpx = -chi * dpx;
    dpy =  chi * dpy;

    if( ( hxl > ZERO ) || ( hyl > ZERO ) || ( hxl < ZERO ) || ( hyl < ZERO ) )
    {
        real_t const delta  = NS(Particles_get_delta_value)( particles, ii );
        real_t const length = NS(MultiPole_get_length)( mp );

        real_t const hxlx   = x * hxl;
        real_t const hyly   = y * hyl;

        NS(Particles_add_to_zeta_value)( particles, ii, chi * ( hyly - hxlx ) );

        dpx += hxl + hxl * delta;
        dpy -= hyl + hyl * delta;

        if( length > ZERO )
        {
            real_t const b1l = chi * NS(MultiPole_get_bal_value)( mp, 0 );
            real_t const a1l = chi * NS(MultiPole_get_bal_value)( mp, 1 );

            dpx -= b1l * hxlx / length;
            dpy += a1l * hyly / length;
        }
    }

    NS(Particles_add_to_px_value)( particles, ii, dpx );
    NS(Particles_add_to_py_value)( particles, ii, dpy );

    return ( SIXTRL_TRACK_RETURN )0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_xy_shift)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(XYShift) *const SIXTRL_RESTRICT xy_shift )
{
    typedef NS(particle_real_t) real_t;

    real_t const minus_dx = -( NS(XYShift_get_dx)( xy_shift ) );
    real_t const minus_dy = -( NS(XYShift_get_dy)( xy_shift ) );

    NS(Particles_add_to_x_value)( particles, ii, minus_dx );
    NS(Particles_add_to_y_value)( particles, ii, minus_dy );

    return 0;
}


SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_srotation)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(SRotation) *const SIXTRL_RESTRICT srotation )
{
    typedef NS(particle_real_t) real_t;

    real_t const sin_z = NS(SRotation_get_sin_angle)( srotation );
    real_t const cos_z = NS(SRotation_get_cos_angle)( srotation );

    real_t const x  = NS(Particles_get_x_value)(  particles, ii );
    real_t const y  = NS(Particles_get_y_value)(  particles, ii );
    real_t const px = NS(Particles_get_px_value)( particles, ii );
    real_t const py = NS(Particles_get_py_value)( particles, ii );

    real_t const x_hat  =  cos_z * x  + sin_z * y;
    real_t const y_hat  = -sin_z * x  + cos_z * y;

    real_t const px_hat =  cos_z * px + sin_z * py;
    real_t const py_hat = -sin_z * px + cos_z * py;

    NS(Particles_set_x_value)(  particles, ii, x_hat );
    NS(Particles_set_y_value)(  particles, ii, y_hat );

    NS(Particles_set_px_value)( particles, ii, px_hat );
    NS(Particles_set_py_value)( particles, ii, py_hat );

    return 0;
}

/* ------------------------------------------------------------------------- */

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_cavity)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(Cavity) *const SIXTRL_RESTRICT cav )
{
    typedef NS(particle_real_t) real_t;

    SIXTRL_STATIC_VAR real_t const PI  =
        ( real_t )3.1415926535897932384626433832795028841971693993751;

    SIXTRL_STATIC_VAR real_t const TWO  = ( real_t )2.0;

    real_t const DEG2RAD  = PI / ( real_t )180.0;
    real_t const K_FACTOR = ( TWO * PI ) / ( real_t )299792458.0;

    real_t const   beta0  = NS(Particles_get_beta0_value)(  particles, ii );
    real_t const   zeta   = NS(Particles_get_zeta_value)(   particles, ii );
    real_t const   chi    = NS(Particles_get_chi_value)(    particles, ii );
    real_t         rvv    = NS(Particles_get_rvv_value)(    particles, ii );
    real_t const   tau    = zeta / ( beta0 * rvv );

    real_t const   phase  = DEG2RAD  * NS(Cavity_get_lag)( cav ) -
                            K_FACTOR * NS(Cavity_get_frequency)( cav ) * tau;

    real_t const energy   = chi * sin( phase ) * NS(Cavity_get_voltage)( cav );

    NS(Particles_add_to_energy_value)( particles, ii, energy );

    return 0;
}


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) index,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(object_type_id_t)    type_id_t;
    typedef NS(buffer_addr_t)       address_t;

    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    type_id_t const    type_id = NS(Object_get_type_id)( be_info );
    address_t const begin_addr = NS(Object_get_begin_addr)( be_info );

    SIXTRL_ASSERT( begin_addr != ( address_t )0u );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( index <  NS(Particles_get_num_of_particles)( p ) );

    switch( type_id )
    {
        case NS(OBJECT_TYPE_DRIFT):
        {
            typedef NS(Drift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_drift)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_DRIFT_EXACT):
        {
            typedef NS(DriftExact)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_drift_exact)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_MULTIPOLE):
        {
            typedef NS(MultiPole)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_multipole)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_CAVITY):
        {
            typedef NS(Cavity)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_cavity)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_XYSHIFT):
        {
            typedef NS(XYShift)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_xy_shift)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_SROTATION):
        {
            typedef NS(SRotation)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_srotation)( p, index, belem );
            break;
        }

        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            typedef NS(BeamBeam4D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_4d)( p, index, belem );
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_6D):
        {
            typedef NS(BeamBeam6D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            ret = NS(Track_particle_beam_beam_6d)( p, index, belem );
            break;
        }

        #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */


        default:
        {
            ret = -8;
        }
    };

    return ret;
}


SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_end )
{
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* be_it = be_begin;
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    for( ; be_it != be_end ; ++be_it )
    {
        ret |= NS(Track_particle_beam_element_obj)( particles, ii, be_it );
    }

    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN
NS(Track_particle_subset_beam_element_obj)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) index,
    NS(particle_num_elements_t) const index_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info )
{
    typedef NS(object_type_id_t)    type_id_t;
    typedef NS(buffer_addr_t)       address_t;

    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    type_id_t const    type_id = NS(Object_get_type_id)( be_info );
    address_t const begin_addr = NS(Object_get_begin_addr)( be_info );

    SIXTRL_ASSERT( begin_addr != ( address_t )0u );
    SIXTRL_ASSERT( index >= ( NS(particle_num_elements_t) )0 );
    SIXTRL_ASSERT( index <= index_end );
    SIXTRL_ASSERT( index_end <= NS(Particles_get_num_of_particles)( p ) );

    switch( type_id )
    {
        case NS(OBJECT_TYPE_DRIFT):
        {
            typedef NS(Drift) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;

            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_drift)( p, index, belem );
            }

            break;
        }

        case NS(OBJECT_TYPE_DRIFT_EXACT):
        {
            typedef NS(DriftExact) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;

            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_drift_exact)( p, index, belem );
            }

            break;
        }

        case NS(OBJECT_TYPE_MULTIPOLE):
        {
            typedef NS(MultiPole) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;

            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_multipole)( p, index, belem );
            }

            break;
        }

        case NS(OBJECT_TYPE_CAVITY):
        {
            typedef NS(Cavity) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;

            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_cavity)( p, index, belem );
            }

            break;
        }

        case NS(OBJECT_TYPE_XYSHIFT):
        {
            typedef NS(XYShift) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;

            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_xy_shift)( p, index, belem );
            }

            break;
        }

        case NS(OBJECT_TYPE_SROTATION):
        {
            typedef NS(SRotation) belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;

            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_srotation)( p, index, belem );
            }

            break;
        }

        #if !defined( SIXTRL_DISABLE_BEAM_BEAM )

        case NS(OBJECT_TYPE_BEAM_BEAM_4D):
        {
            typedef NS(BeamBeam4D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_beam_beam_4d)( p, index, belem );
            }
            break;
        }

        case NS(OBJECT_TYPE_BEAM_BEAM_6D):
        {
            typedef NS(BeamBeam6D)   belem_t;
            typedef SIXTRL_BE_ARGPTR_DEC belem_t const* ptr_to_belem_t;
            ptr_to_belem_t belem = ( ptr_to_belem_t )( uintptr_t )begin_addr;

            for( ; index < index_end ; ++index )
            {
                ret |= NS(Track_particle_beam_beam_6d)( p, index, belem );
            }
            break;
        }

        #endif /* !defined( SIXTRL_DISABLE_BEAM_BEAM ) */

        default:
        {
            ret = -1;
        }
    };

    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    NS(particle_num_elements_t) p_index_begin,
    NS(particle_num_elements_t) const p_index_end,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_it,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_end )
{
    SIXTRL_TRACK_RETURN ret = ( SIXTRL_TRACK_RETURN )0;

    SIXTRL_ASSERT( ( ( uintptr_t )be_info_end ) >= ( uintptr_t )be_info_it );

    SIXTRL_ASSERT( ( be_info_it != SIXTRL_NULLPTR ) ||
                   ( be_info_it == SIXTRL_NULLPTR ) );

    for( ; be_info_it != be_info_end ; ++be_info_it )
    {
        ret |= NS(Track_particle_subset_beam_element_obj)(
            p, p_index_begin, p_index_end, be_info_it );
    }

    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_beam_element_objs)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT p,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_begin,
    SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* SIXTRL_RESTRICT be_info_end )
{
    typedef NS(particle_num_elements_t) num_elem_t;

    return NS(Track_particle_subset_beam_element_objs)( p,
        ( num_elem_t )0u, NS(Particles_get_num_of_particles)( p ),
        be_info_begin, be_info_end );
}

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE )

struct NS(Buffer);

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const beam_elements )
{
    return NS(Track_particle_subset_beam_element_objs)(
        particles, ( NS(particle_num_elements_t) )0u,
        NS(Particles_get_num_of_particles)( particles ),
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particles_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const beam_elements,
    NS(buffer_size_t) const beam_element_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_index <
        NS(Buffer_get_num_of_objects)( beam_elements ) );

    ptr_be_obj = ptr_be_obj + beam_element_index;

    return NS(Track_particle_subset_beam_element_obj)(
        particles, ( NS(particle_num_elements_t) )0u,
        NS(Particles_get_num_of_particles)( particles ), ptr_be_obj );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_index <
        NS(Buffer_get_num_of_objects)( beam_elements ) );

    ptr_be_obj = ptr_be_obj + beam_element_index;

    return NS(Track_particle_beam_element_obj)(
        particles, particle_index, ptr_be_obj );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_element_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_begin_index,
    NS(buffer_size_t) const beam_element_end_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t be_end   = SIXTRL_NULLPTR;
    ptr_obj_t be_begin = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_begin_index <= beam_element_end_index );
    SIXTRL_ASSERT( beam_element_end_index <=
                   NS(Buffer_get_num_of_objects)( beam_elements ) );

    be_end   = be_begin + beam_element_end_index;
    be_begin = be_begin + beam_element_begin_index;

    return NS(Track_particle_beam_element_objs)(
        particles, particle_index, be_begin, be_end );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_particle_beam_element_objs)(
        particles, particle_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );

}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_element)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t ptr_be_obj = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( ptr_be_obj != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_index <
        NS(Buffer_get_num_of_objects)( beam_elements ) );

    ptr_be_obj = ptr_be_obj + beam_element_index;

    return NS(Track_particle_subset_beam_element_obj)(
        particles, particle_begin_index, particle_end_index, ptr_be_obj );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_element_subset)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements,
    NS(buffer_size_t) const beam_element_begin_index,
    NS(buffer_size_t) const beam_element_end_index )
{
    typedef SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const* ptr_obj_t;

    ptr_obj_t be_end   = SIXTRL_NULLPTR;
    ptr_obj_t be_begin = NS(Buffer_get_const_objects_begin)( beam_elements );

    SIXTRL_ASSERT( be_begin != SIXTRL_NULLPTR );
    SIXTRL_ASSERT( beam_element_begin_index <= beam_element_end_index );
    SIXTRL_ASSERT( beam_element_end_index <=
                   NS(Buffer_get_num_of_objects)( beam_elements ) );

    be_end   = be_begin + beam_element_end_index;
    be_begin = be_begin + beam_element_begin_index;

    return NS(Track_particle_subset_beam_element_objs)(
        particles, particle_begin_index, particle_end_index, be_begin, be_end );
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_subset_beam_elements)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_begin_index,
    NS(particle_num_elements_t) const particle_end_index,
    SIXTRL_BUFFER_ARGPTR_DEC const struct NS(Buffer) *const
        SIXTRL_RESTRICT beam_elements )
{
    return NS(Track_particle_subset_beam_element_objs)(
        particles, particle_begin_index, particle_end_index,
        NS(Buffer_get_const_objects_begin)( beam_elements ),
        NS(Buffer_get_const_objects_end)( beam_elements ) );
}

#endif /* !defined( _GPUCODE ) */

/* ------------------------------------------------------------------------- */

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_TRACK_H__ */

/* end: sixtracklib/common/track.h */
