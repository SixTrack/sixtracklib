#ifndef SIXTRACKLIB_COMMON_BE_BEAMBEAM_TRACK_BEAMBEAM_H__
#define SIXTRACKLIB_COMMON_BE_BEAMBEAM_TRACK_BEAMBEAM_H__

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/beam_elements_defines.h"
    #include "sixtracklib/common/particles.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

struct NS(BeamBeam4D);
struct NS(BeamBeam6D);

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_4d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT bb );

SIXTRL_FN SIXTRL_STATIC SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_6d)(
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
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/be_beambeam/be_beambeam4d.h"
    #include "sixtracklib/common/be_beambeam/be_beambeam6d.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined( _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_4d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t) const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam4D) *const SIXTRL_RESTRICT bb )
{
    SIXTRL_TRACK_RETURN ret = 0;

    typedef NS(beambeam4d_real_const_ptr_t)  bb_data_ptr_t;

    SIXTRL_UINT64_T const data_size = NS(BeamBeam4D_get_data_size)( bb );
    bb_data_ptr_t data = NS(BeamBeam4D_get_const_data)( bb );
    (void) data_size; // just to avoid error: unused variable

    SIXTRL_REAL_T x = NS(Particles_get_x_value)( particles, particle_index );
    x += ( SIXTRL_REAL_T )0.0 + 0.*data[0];
    printf("BB4D data[0]%.2e\n", data[0]);

    NS(Particles_set_x_value)( particles, particle_index, x );

    return ret;
}

SIXTRL_INLINE SIXTRL_TRACK_RETURN NS(Track_particle_beam_beam_6d)(
    SIXTRL_PARTICLE_ARGPTR_DEC NS(Particles)* SIXTRL_RESTRICT particles,
    NS(particle_num_elements_t)  const particle_index,
    SIXTRL_BE_ARGPTR_DEC const struct NS(BeamBeam6D) *const SIXTRL_RESTRICT bb )
{
    typedef NS(beambeam6d_real_const_ptr_t)  bb_data_ptr_t;

    SIXTRL_TRACK_RETURN ret = 0;

    SIXTRL_UINT64_T const data_size = NS(BeamBeam6D_get_data_size)( bb ); 
    (void) data_size; // just to avoid error: unused variable
    bb_data_ptr_t data = NS(BeamBeam6D_get_const_data)( bb );

    int i_slice;

    // Start Gianni's part
    BB6D_data* bb6ddata = (BB6D_data*) data;

    // Get pointers
    double* N_part_per_slice = (double*)(((uint64_t*) (&(bb6ddata->N_part_per_slice))) + ((uint64_t) bb6ddata->N_part_per_slice) + 1);
    double* x_slices_star = (double*)(((uint64_t*) (&(bb6ddata->x_slices_star))) + ((uint64_t) bb6ddata->x_slices_star) + 1);
    double* y_slices_star = (double*)(((uint64_t*) (&(bb6ddata->y_slices_star))) + ((uint64_t) bb6ddata->y_slices_star) + 1);
    double* sigma_slices_star = (double*)(((uint64_t*) (&(bb6ddata->sigma_slices_star))) + ((uint64_t) bb6ddata->sigma_slices_star) + 1);
    


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

    SIXTRL_REAL_T x = NS(Particles_get_x_value)( particles, particle_index );
    SIXTRL_REAL_T px = NS(Particles_get_px_value)( particles, particle_index );
    SIXTRL_REAL_T y = NS(Particles_get_y_value)( particles, particle_index );
    SIXTRL_REAL_T py = NS(Particles_get_py_value)( particles, particle_index );
    SIXTRL_REAL_T zeta = NS(Particles_get_zeta_value)( particles, particle_index );
    SIXTRL_REAL_T delta = NS(Particles_get_delta_value)( particles, particle_index );

    SIXTRL_REAL_T q0 = NS(Particles_get_q0_value)( particles, particle_index );
    SIXTRL_REAL_T P0 = NS(Particles_get_p0c_value)( particles, particle_index ); // eV

    P0 = P0/C_LIGHT*QELEM;

    // Change reference frame
    double x_star =     x     - bb6ddata->x_CO    - bb6ddata->delta_x;
    double px_star =    px    - bb6ddata->px_CO;
    double y_star =     y     - bb6ddata->y_CO    - bb6ddata->delta_y;
    double py_star =    py    - bb6ddata->py_CO;
    double sigma_star = zeta  - bb6ddata->sigma_CO;
    double delta_star = delta - bb6ddata->delta_CO;

    // Boost coordinates of the weak beam
    BB6D_boost(&(bb6ddata->parboost), &x_star, &px_star, &y_star, &py_star, 
                &sigma_star, &delta_star);


    // Synchro beam
    for (i_slice=0; i_slice<N_slices; i_slice++)
    {
        double sigma_slice_star = sigma_slices_star[i_slice];
        double x_slice_star = x_slices_star[i_slice];
        double y_slice_star = y_slices_star[i_slice];
        
        //Compute force scaling factor
        double Ksl = N_part_per_slice[i_slice]*bb6ddata->q_part*q0/(P0*C_LIGHT);

        //Identify the Collision Point (CP)
        double S = 0.5*(sigma_star - sigma_slice_star);
        
        // Propagate sigma matrix
        double Sig_11_hat_star, Sig_33_hat_star, costheta, sintheta;
        double dS_Sig_11_hat_star, dS_Sig_33_hat_star, dS_costheta, dS_sintheta;
        
        // Get strong beam shape at the CP
        BB6D_propagate_Sigma_matrix(&(bb6ddata->Sigmas_0_star),
            S, bb6ddata->threshold_singular, 1,
            &Sig_11_hat_star, &Sig_33_hat_star, 
            &costheta, &sintheta,
            &dS_Sig_11_hat_star, &dS_Sig_33_hat_star, 
            &dS_costheta, &dS_sintheta);
            
        // Evaluate transverse coordinates of the weake baem w.r.t. the strong beam centroid
        double x_bar_star = x_star + px_star*S - x_slice_star;
        double y_bar_star = y_star + py_star*S - y_slice_star;
        
        // Move to the uncoupled reference frame
        double x_bar_hat_star = x_bar_star*costheta +y_bar_star*sintheta;
        double y_bar_hat_star = -x_bar_star*sintheta +y_bar_star*costheta;
        
        // Compute derivatives of the transformation
        double dS_x_bar_hat_star = x_bar_star*dS_costheta +y_bar_star*dS_sintheta;
        double dS_y_bar_hat_star = -x_bar_star*dS_sintheta +y_bar_star*dS_costheta;
        
        // Get transverse fieds
        double Ex, Ey, Gx, Gy;
        get_Ex_Ey_Gx_Gy_gauss(x_bar_hat_star, y_bar_hat_star, 
            sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), bb6ddata->min_sigma_diff,
            &Ex, &Ey, &Gx, &Gy);
            
        // Compute kicks
        double Fx_hat_star = Ksl*Ex;
        double Fy_hat_star = Ksl*Ey;
        double Gx_hat_star = Ksl*Gx;
        double Gy_hat_star = Ksl*Gy;
        
        // Move kisks to coupled reference frame
        double Fx_star = Fx_hat_star*costheta - Fy_hat_star*sintheta;
        double Fy_star = Fx_hat_star*sintheta + Fy_hat_star*costheta;
        
        // Compute longitudinal kick
        double Fz_star = 0.5*(Fx_hat_star*dS_x_bar_hat_star  + Fy_hat_star*dS_y_bar_hat_star+
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
    zeta = sigma_star + bb6ddata->sigma_CO                    - bb6ddata->Dsigma_sub;
    delta = delta_star + bb6ddata->delta_CO                    - bb6ddata->Ddelta_sub;


    x += ( SIXTRL_REAL_T )0.0 + 0.*data[0];
    printf("BB6D data[0]%.2e\n", data[0]);
    NS(Particles_set_x_value)( particles, particle_index, x );

    // Debug
    (void) N_slices;
    (void) N_part_per_slice;
    (void) x_slices_star;
    (void) y_slices_star;   
    (void) sigma_slices_star;
    (void) x_star;
    (void) px_star;
    (void) y_star;
    (void) py_star;
    (void) sigma_star;
    (void) delta_star;
    // End Gianni's part

    return ret;
}

#if !defined( _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */
#endif 
/* end: sixtracklib/common/be_beambeam/track_beambeam.h */
