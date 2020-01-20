#ifndef SIXTRACKLIB_COMMON_BE_RFMULTIPOLE_TRACK_C99_H__
#define SIXTRACKLIB_COMMON_BE_RFMULTIPOLE_TRACK_C99_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/track/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(RFMultiPole);

SIXTRL_STATIC SIXTRL_FN NS(track_status_t) NS(Track_particle_rf_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(RFMultiPole)
        *const SIXTRL_RESTRICT mpole );

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/be_rfmultipole/be_rfmultipole.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE NS(track_status_t) NS(Track_particle_rf_multipole)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const ii,
    SIXTRL_BE_ARGPTR_DEC const NS(RFMultiPole) *const SIXTRL_RESTRICT mpole )
{
    typedef NS(rf_multipole_real_t) real_t;
    typedef NS(rf_multipole_int_t) int_t;

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, ii ) ==
                   ( NS(particle_index_t) )1 );

    SIXTRL_BUFFER_DATAPTR_DEC NS(rf_multipole_real_t) const* bal =
        NS(RFMultiPole_const_bal)( mpole );

    SIXTRL_BUFFER_DATAPTR_DEC NS(rf_multipole_real_t) const* phase =
        NS(RFMultiPole_const_phase)( mpole );

    real_t const order = NS(RFMultiPole_order)( mpole );

    real_t const voltage = NS(RFMultiPole_voltage)( mpole );
    real_t const frequency = NS(RFMultiPole_frequency)( mpole );
    real_t const lag = NS(RFMultiPole_lag)( mpole );

    real_t const DEG2RAD  = SIXTRL_PI / ( real_t )180.0;
    real_t const K_FACTOR = ( ( real_t )2.0 * SIXTRL_PI ) / SIXTRL_C_LIGHT;
    real_t const k = K_FACTOR * frequency;

    real_t const x = NS(Particles_get_x_value)( particles, ii );
    real_t const y = NS(Particles_get_y_value)( particles, ii );

    real_t const beta0  = NS(Particles_get_beta0_value)( particles, ii );
    real_t const zeta   = NS(Particles_get_zeta_value)( particles, ii );
    real_t const q      = NS(Particles_get_q0_value)( particles, ii ) *
                          NS(Particles_get_charge_ratio_value)( particles, ii );
    real_t const rvv    = NS(Particles_get_rvv_value)( particles, ii );
    real_t const tau    = zeta / ( beta0 * rvv );

    real_t const ktau = k * tau;

    real_t dpx = 0.0;
    real_t dpy = 0.0;
    real_t dptr = 0.0;
    real_t zre = 1.0;
    real_t zim = 0.0;

    for (int_t kk = 0; kk <= order; kk++)
    {
        real_t const pn_kk = DEG2RAD * phase[2*kk] - ktau;
        real_t const ps_kk = DEG2RAD * phase[2*kk+1] - ktau;

        real_t const bal_n_kk = bal[2*kk];
        real_t const bal_s_kk = bal[2*kk+1];

        real_t const cn = cos(pn_kk);
        real_t const cs = cos(ps_kk);
        real_t const sn = sin(pn_kk);
        real_t const ss = sin(ps_kk);

        dpx += cn * (bal_n_kk * zre) - cs * (bal_s_kk * zim);
        dpy += cs * (bal_s_kk * zre) + cn * (bal_n_kk * zim);

        real_t const zret = zre * x - zim * y;
        zim = zim * x + zre * y;
        zre = zret;

        dptr += sn * (bal_n_kk * zre) - ss * (bal_s_kk * zim);
    }

    real_t const cav_energy = q * ( voltage * sin(lag * DEG2RAD - ktau) );
    real_t const p0c = NS(Particles_get_p0c_value)( particles, ii);
    real_t const rfmultipole_energy = - q * ( (k * p0c) * dptr );

    real_t const chi    = NS(Particles_get_chi_value)( particles, ii );

    real_t const px_kick = - chi * dpx;
    real_t const py_kick =   chi * dpy;
    real_t const energy_kick = cav_energy + rfmultipole_energy;

    NS(Particles_add_to_px_value)( particles, ii, px_kick );
    NS(Particles_add_to_py_value)( particles, ii, py_kick );
    NS(Particles_add_to_energy_value)( particles, ii, energy_kick );

    return SIXTRL_TRACK_SUCCESS;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_BE_RFMULTIPOLE_TRACK_C99_H__ */
