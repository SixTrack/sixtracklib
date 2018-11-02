#ifndef SIXTRACKLIB_COMMON_BE_BEAMBEAM_TRACK_BEAMBEAM_H__
#define SIXTRACKLIB_COMMON_BE_BEAMBEAM_TRACK_BEAMBEAM_H__

#ifndef SIXTRL_BB6_GET_PTR
	#define SIXTRL_BB_GET_PTR(dataptr,name) \
		 (SIXTRL_BE_DATAPTR_DEC real_t*)(((SIXTRL_BE_DATAPTR_DEC u64_t*) (&((dataptr)->name))) \
		 	+ ((u64_t) (dataptr)->name) + 1)
#endif

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/internal/beam_elements_defines.h"
    #include "sixtracklib/common/internal/objects_type_id.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(BeamBeam4D);
struct NS(BeamBeam6D);

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_beam_4d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT bb );

SIXTRL_FN SIXTRL_STATIC int NS(Track_particle_beam_beam_6d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t)  const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT bb );

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
    #include "sixtracklib/common/constants.h"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/be_beambeam/be_beambeam4d.h"
    #include "sixtracklib/common/be_beambeam/be_beambeam6d.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE int NS(Track_particle_beam_beam_4d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_idx,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT bb )
{
    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;
    //typedef SIXTRL_UINT64_T u64_t;
    typedef SIXTRL_REAL_T real_t;
    typedef SIXTRL_BE_DATAPTR_DEC BB4D_data* BB4D_data_ptr_t;

    bb_data_ptr_t data = NS(BeamBeam4D_get_const_data)( bb );

    BB4D_data_ptr_t bb4ddata = (BB4D_data_ptr_t) data;

    /*
    // Test data transfer
    printf("4D: q_part = %e\n",bb4ddata->q_part);
    printf("4D: N_part = %e\n",bb4ddata->N_part);
    printf("4D: sigma_x = %e\n",bb4ddata->sigma_x);
    printf("4D: sigma_y = %e\n",bb4ddata->sigma_y);
    printf("4D: beta_s = %e\n",bb4ddata->beta_s);
    printf("4D: min_sigma_diff = %e\n",bb4ddata->min_sigma_diff);
    printf("4D: Delta_x = %e\n",bb4ddata->Delta_x);
    printf("4D: Delta_y = %e\n",bb4ddata->Delta_y);
    printf("4D: Dpx_sub = %e\n",bb4ddata->Dpx_sub);
    printf("4D: Dpy_sub = %e\n",bb4ddata->Dpy_sub);
    printf("4D: enabled = %ld\n",bb4ddata->enabled);
    */

    SIXTRL_ASSERT( NS(Particles_get_num_of_particles)( particles ) >
                   particle_idx );

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, particle_idx ) ==
                   ( NS(particle_index_t) )1 );


    if (bb4ddata->enabled) {

        real_t const charge = NS(Particles_get_q_value)( particles, particle_idx );

        real_t const x =
            NS(Particles_get_x_value)( particles, particle_idx ) - bb4ddata->Delta_x;

        real_t const y =
            NS(Particles_get_y_value)( particles, particle_idx ) - bb4ddata->Delta_y;

        real_t const chi = NS(Particles_get_chi_value)( particles, particle_idx );

        real_t const beta =
            NS(Particles_get_beta0_value)( particles, particle_idx ) /
            NS(Particles_get_rvv_value)( particles, particle_idx );

        real_t const p0c =
            NS(Particles_get_p0c_value)( particles, particle_idx )*SIXTRL_QELEM;

        real_t Ex, Ey, Gx, Gy;
        get_Ex_Ey_Gx_Gy_gauss(x, y, bb4ddata->sigma_x, bb4ddata->sigma_y,
                bb4ddata->min_sigma_diff, 1,
                &Ex, &Ey, &Gx, &Gy);

        real_t fact_kick = chi * bb4ddata->N_part * bb4ddata->q_part * charge * \
            (1. + beta * bb4ddata->beta_s)/(p0c*(beta + bb4ddata->beta_s));

        NS(Particles_add_to_px_value)(
            particles, particle_idx, fact_kick*Ex - bb4ddata->Dpx_sub );

        NS(Particles_add_to_py_value)(
            particles, particle_idx, fact_kick*Ey - bb4ddata->Dpy_sub );
    }

    return 0;
}

SIXTRL_INLINE int NS(Track_particle_beam_beam_6d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t)  const particle_idx,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT bb )
{
    typedef NS(beambeam6d_real_const_ptr_t)  bb_data_ptr_t;
    typedef SIXTRL_UINT64_T u64_t;
    typedef SIXTRL_REAL_T real_t;
    typedef SIXTRL_BE_DATAPTR_DEC BB6D_data* BB6D_data_ptr_t;

    int i_slice;

    bb_data_ptr_t data = NS(BeamBeam6D_get_const_data)( bb );

    // Start Gianni's part
    BB6D_data_ptr_t bb6ddata = (BB6D_data_ptr_t) data;

    SIXTRL_ASSERT( NS(Particles_get_num_of_particles)( particles ) >
                   particle_idx );

    SIXTRL_ASSERT( NS(Particles_get_state_value)( particles, particle_idx ) ==
                   ( NS(particle_index_t) )1 );

    if (bb6ddata->enabled) {

        // Get pointers
        SIXTRL_BE_DATAPTR_DEC real_t* N_part_per_slice = SIXTRL_BB_GET_PTR(bb6ddata, N_part_per_slice);
		SIXTRL_BE_DATAPTR_DEC real_t* x_slices_star = SIXTRL_BB_GET_PTR(bb6ddata, x_slices_star);
		SIXTRL_BE_DATAPTR_DEC real_t* y_slices_star = SIXTRL_BB_GET_PTR(bb6ddata, y_slices_star);
		SIXTRL_BE_DATAPTR_DEC real_t* sigma_slices_star = SIXTRL_BB_GET_PTR(bb6ddata, sigma_slices_star);


        int N_slices = (int)(bb6ddata->N_slices);

        /*
        // Check data transfer
        printf("sphi=%e\n",(bb6ddata->parboost).sphi);
        printf("calpha=%e\n",(bb6ddata->parboost).calpha);
        printf("S33=%e\n",(bb6ddata->Sigmas_0_star).Sig_33_0);
        printf("N_slices=%d\n",N_slices);
        printf("N_part_per_slice[0]=%e\n",N_part_per_slice[0]);
        printf("N_part_per_slice[1]=%e\n",N_part_per_slice[1]);
        printf("x_slices_star[0]=%e\n",x_slices_star[0]);
        printf("x_slices_star[1]=%e\n",x_slices_star[1]);
        printf("y_slices_star[0]=%e\n",y_slices_star[0]);
        printf("y_slices_star[1]=%e\n",y_slices_star[1]);
        printf("sigma_slices_star[0]=%e\n",sigma_slices_star[0]);
        printf("sigma_slices_star[1]=%e\n",sigma_slices_star[1]);
        */

        real_t x = NS(Particles_get_x_value)( particles, particle_idx );
        real_t px = NS(Particles_get_px_value)( particles, particle_idx );
        real_t y = NS(Particles_get_y_value)( particles, particle_idx );
        real_t py = NS(Particles_get_py_value)( particles, particle_idx );
        real_t zeta = NS(Particles_get_zeta_value)( particles, particle_idx );
        real_t delta = NS(Particles_get_delta_value)( particles, particle_idx );

        real_t q0 = SIXTRL_QELEM*NS(Particles_get_q0_value)( particles, particle_idx );
        real_t p0c = NS(Particles_get_p0c_value)( particles, particle_idx ); // eV

        real_t P0 = p0c/SIXTRL_C_LIGHT*SIXTRL_QELEM;

        // Change reference frame
        real_t x_star =     x     - bb6ddata->x_CO    - bb6ddata->delta_x;
        real_t px_star =    px    - bb6ddata->px_CO;
        real_t y_star =     y     - bb6ddata->y_CO    - bb6ddata->delta_y;
        real_t py_star =    py    - bb6ddata->py_CO;
        real_t sigma_star = zeta  - bb6ddata->sigma_CO;
        real_t delta_star = delta - bb6ddata->delta_CO;

        // Boost coordinates of the weak beam
        BB6D_boost(&(bb6ddata->parboost), &x_star, &px_star, &y_star, &py_star,
                    &sigma_star, &delta_star);


        // Synchro beam
        for (i_slice=0; i_slice<N_slices; i_slice++)
        {
            real_t sigma_slice_star = sigma_slices_star[i_slice];
            real_t x_slice_star = x_slices_star[i_slice];
            real_t y_slice_star = y_slices_star[i_slice];

            //Compute force scaling factor
            real_t Ksl = N_part_per_slice[i_slice]*bb6ddata->q_part*q0/(P0* SIXTRL_C_LIGHT);

            //Identify the Collision Point (CP)
            real_t S = 0.5*(sigma_star - sigma_slice_star);

            // Propagate sigma matrix
            real_t Sig_11_hat_star, Sig_33_hat_star, costheta, sintheta;
            real_t dS_Sig_11_hat_star, dS_Sig_33_hat_star, dS_costheta, dS_sintheta;

            // Get strong beam shape at the CP
            BB6D_propagate_Sigma_matrix(&(bb6ddata->Sigmas_0_star),
                S, bb6ddata->threshold_singular, 1,
                &Sig_11_hat_star, &Sig_33_hat_star,
                &costheta, &sintheta,
                &dS_Sig_11_hat_star, &dS_Sig_33_hat_star,
                &dS_costheta, &dS_sintheta);

            // Evaluate transverse coordinates of the weake baem w.r.t. the strong beam centroid
            real_t x_bar_star = x_star + px_star*S - x_slice_star;
            real_t y_bar_star = y_star + py_star*S - y_slice_star;

            // Move to the uncoupled reference frame
            real_t x_bar_hat_star = x_bar_star*costheta +y_bar_star*sintheta;
            real_t y_bar_hat_star = -x_bar_star*sintheta +y_bar_star*costheta;

            // Compute derivatives of the transformation
            real_t dS_x_bar_hat_star = x_bar_star*dS_costheta +y_bar_star*dS_sintheta;
            real_t dS_y_bar_hat_star = -x_bar_star*dS_sintheta +y_bar_star*dS_costheta;

            // Get transverse fieds
            real_t Ex, Ey, Gx, Gy;
            get_Ex_Ey_Gx_Gy_gauss(x_bar_hat_star, y_bar_hat_star,
                sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), bb6ddata->min_sigma_diff, 0,
                &Ex, &Ey, &Gx, &Gy);

            // Compute kicks
            real_t Fx_hat_star = Ksl*Ex;
            real_t Fy_hat_star = Ksl*Ey;
            real_t Gx_hat_star = Ksl*Gx;
            real_t Gy_hat_star = Ksl*Gy;

            // Move kisks to coupled reference frame
            real_t Fx_star = Fx_hat_star*costheta - Fy_hat_star*sintheta;
            real_t Fy_star = Fx_hat_star*sintheta + Fy_hat_star*costheta;

            // Compute longitudinal kick
            real_t Fz_star = 0.5*(Fx_hat_star*dS_x_bar_hat_star  + Fy_hat_star*dS_y_bar_hat_star+
                           Gx_hat_star*dS_Sig_11_hat_star + Gy_hat_star*dS_Sig_33_hat_star);

            // Apply the kicks (Hirata's synchro-beam)
            delta_star = delta_star + Fz_star+0.5*(
                        Fx_star*(px_star+0.5*Fx_star)+
                        Fy_star*(py_star+0.5*Fy_star));
            x_star = x_star - S*Fx_star;
            px_star = px_star + Fx_star;
            y_star = y_star - S*Fy_star;
            py_star = py_star + Fy_star;


        }

        // Inverse boost on the coordinates of the weak beam
        BB6D_inv_boost(&(bb6ddata->parboost), &x_star, &px_star, &y_star, &py_star,
                    &sigma_star, &delta_star);


        // Go back to original reference frame and remove dipolar effect
        x =     x_star     + bb6ddata->x_CO   + bb6ddata->delta_x  - bb6ddata->Dx_sub;
        px =    px_star    + bb6ddata->px_CO                       - bb6ddata->Dpx_sub;
        y =     y_star     + bb6ddata->y_CO   + bb6ddata->delta_y  - bb6ddata->Dy_sub;
        py =    py_star    + bb6ddata->py_CO                       - bb6ddata->Dpy_sub;
        zeta =  sigma_star + bb6ddata->sigma_CO                    - bb6ddata->Dsigma_sub;
        delta = delta_star + bb6ddata->delta_CO                    - bb6ddata->Ddelta_sub;


        NS(Particles_set_x_value)( particles, particle_idx, x );
        NS(Particles_set_px_value)( particles, particle_idx, px );
        NS(Particles_set_y_value)( particles, particle_idx, y );
        NS(Particles_set_py_value)( particles, particle_idx, py );
        NS(Particles_set_zeta_value)( particles, particle_idx, zeta );
        NS(Particles_update_delta_value)( particles, particle_idx, delta );
    }


    return 0;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */


#endif /* SIXTRACKLIB_COMMON_BE_BEAMBEAM_TRACK_BEAMBEAM_H__ */

/* end: sixtracklib/common/be_beambeam/track_beambeam.h */
