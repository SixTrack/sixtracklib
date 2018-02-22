import propagate_sigma_matrix as psm
import boost as boost
import slicing
import transverse_efields as tef


import numpy as np

from scipy.constants import c as c_light

def int_to_float64arr(val):
    temp = np.zeros(1, (np.float64, {'i64':('i8',0)}))
    temp['i64'][0] = val
    return temp


class BB6D_Data(object):
    def __init__(self, q_part,
            parboost, Sigmas_0_star, N_slices, N_part_per_slice,
            x_slices_star, y_slices_star, sigma_slices_star,
            min_sigma_diff, threshold_singular):
                
        self.q_part = q_part
        self.parboost = parboost
        self.Sigmas_0_star = Sigmas_0_star
        self.min_sigma_diff = min_sigma_diff
        self.threshold_singular = threshold_singular
        self.N_slices = N_slices
        self.N_part_per_slice = N_part_per_slice
        self.x_slices_star = x_slices_star
        self.y_slices_star = y_slices_star
        self.sigma_slices_star = sigma_slices_star
    
    def tobuffer(self):

        buffer_list = []
        # Buffers corresponding to BB6D struct
        buffer_list.append(np.array([self.q_part], dtype=np.float64))
        buffer_list.append(self.parboost.tobuffer())
        buffer_list.append(self.Sigmas_0_star.tobuffer())
        buffer_list.append(np.array([self.min_sigma_diff], dtype=np.float64))
        buffer_list.append(np.array([self.threshold_singular], dtype=np.float64))
        buffer_list.append(int_to_float64arr(self.N_slices))
        buffer_list.append(int_to_float64arr(3))# offset to N_part_per_slice
        buffer_list.append(int_to_float64arr(2+self.N_slices))# offset to x_slices_star
        buffer_list.append(int_to_float64arr(1+2*self.N_slices))# offset to y_slices_star
        buffer_list.append(int_to_float64arr(0+3*self.N_slices))# offset to sigma_slices_star

        # Buffers corresponding to arrays
        buffer_list.append(np.array(self.N_part_per_slice, dtype=np.float64))
        buffer_list.append(np.array(self.x_slices_star, dtype=np.float64))
        buffer_list.append(np.array(self.y_slices_star, dtype=np.float64))
        buffer_list.append(np.array(self.sigma_slices_star, dtype=np.float64))

        buf = np.concatenate(buffer_list)
        
        return buf
        
        
        
        
def BB6D_init(q_part, N_part_tot, sigmaz, N_slices, min_sigma_diff, threshold_singular,
                phi, alpha, 
                Sig_11_0, Sig_12_0, Sig_13_0, 
                Sig_14_0, Sig_22_0, Sig_23_0, 
                Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0):
                    
    # Prepare data for Lorentz transformation
    parboost = boost.ParBoost(phi=phi, alpha=alpha)

    # Prepare data with strong beam shape
    Sigmas_0 = psm.Sigmas(Sig_11_0, Sig_12_0, Sig_13_0, 
                        Sig_14_0, Sig_22_0, Sig_23_0, 
                        Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0)
                        
    # Boost strong beam shape
    Sigmas_0_star = psm.boost_sigmas(Sigmas_0, parboost.cphi)

    # Generate info about slices
    z_centroids, _, N_part_per_slice = slicing.constant_charge_slicing_gaussian(N_part_tot, sigmaz, N_slices)

    # Sort according to z, head at the first position in the arrays
    ind_sorted = np.argsort(z_centroids)[::-1]
    z_centroids = np.take(z_centroids, ind_sorted)
    N_part_per_slice = np.take(N_part_per_slice, ind_sorted)

    # By boosting the strong z and all zeros, I get the transverse coordinates of the strong beam in the ref system of the weak
    boost_vect = np.vectorize(boost.boost, excluded='parboost')
    x_slices_star, px_slices_star, y_slices_star, py_slices_star, sigma_slices_star, delta_slices_star = boost_vect(x=0*z_centroids, px=0*z_centroids, 
                        y=0*z_centroids, py=0*z_centroids, sigma=z_centroids, delta=0*z_centroids, parboost=parboost)
                   
    bb6d_data = BB6D_Data(q_part, parboost, Sigmas_0_star, N_slices, 
       N_part_per_slice, x_slices_star, y_slices_star, sigma_slices_star, min_sigma_diff, threshold_singular)
                
       
    return bb6d_data
    
def BB6D_track(x, px, y, py, sigma, delta, q0, p0, bb6ddata):

    q_part = bb6ddata.q_part
    parboost = bb6ddata.parboost
    Sigmas_0_star = bb6ddata.Sigmas_0_star
    N_slices = bb6ddata.N_slices
    N_part_per_slice = bb6ddata.N_part_per_slice
    x_slices_star = bb6ddata.x_slices_star
    y_slices_star = bb6ddata.y_slices_star
    sigma_slices_star = bb6ddata.sigma_slices_star
    min_sigma_diff = bb6ddata.min_sigma_diff
    threshold_singular = bb6ddata.threshold_singular
    
    # Boost coordinates of the weak beam
    x_star, px_star, y_star, py_star, sigma_star, delta_star = boost.boost(x, px, y, py, sigma, delta, parboost)
    #~ x_star, px_star, y_star, py_star, sigma_star, delta_star = (x, px, y, py, sigma, delta)
    for i_slice in range(N_slices):
        sigma_slice_star = sigma_slices_star[i_slice]
        x_slice_star = x_slices_star[i_slice]
        y_slice_star = y_slices_star[i_slice]
        
        # Compute force scaling factor
        Ksl = N_part_per_slice[i_slice]*q_part*q0/(p0*c_light)
        
        # Identify the Collision Point (CP)
        S = 0.5*(sigma_star - sigma_slice_star)
        
        # Get strong beam shape at the CP
        Sig_11_hat_star, Sig_33_hat_star, costheta, sintheta,\
            dS_Sig_11_hat_star, dS_Sig_33_hat_star, dS_costheta, dS_sintheta,\
            extra_data = psm.propagate_Sigma_matrix(Sigmas_0_star, S, threshold_singular=threshold_singular)
            
        # Evaluate transverse coordinates of the weake baem w.r.t. the strong beam centroid
        x_bar_star = x_star + px_star*S - x_slice_star
        y_bar_star = y_star + py_star*S - y_slice_star
        
        # Move to the uncoupled reference frame
        x_bar_hat_star = x_bar_star*costheta +y_bar_star*sintheta
        y_bar_hat_star = -x_bar_star*sintheta +y_bar_star*costheta
        
        # Compute derivatives of the transformation
        dS_x_bar_hat_star = x_bar_star*dS_costheta +y_bar_star*dS_sintheta
        dS_y_bar_hat_star = -x_bar_star*dS_sintheta +y_bar_star*dS_costheta
        
        # Compute normalized field
        Ex, Ey, Gx, Gy = tef.get_Ex_Ey_Gx_Gy_gauss(x=x_bar_hat_star, y=y_bar_hat_star, 
                            sigma_x=np.sqrt(Sig_11_hat_star), sigma_y=np.sqrt(Sig_33_hat_star),
                            min_sigma_diff = min_sigma_diff)
                            
        # Compute kicks
        Fx_hat_star = Ksl*Ex
        Fy_hat_star = Ksl*Ey
        Gx_hat_star = Ksl*Gx
        Gy_hat_star = Ksl*Gy
        
        # Move kisks to coupled reference frame
        Fx_star = Fx_hat_star*costheta - Fy_hat_star*sintheta
        Fy_star = Fx_hat_star*sintheta + Fy_hat_star*costheta
        
        # Compute longitudinal kick
        Fz_star = 0.5*(Fx_hat_star*dS_x_bar_hat_star  + Fy_hat_star*dS_y_bar_hat_star+\
                       Gx_hat_star*dS_Sig_11_hat_star + Gy_hat_star*dS_Sig_33_hat_star)
                       
        # Apply the kicks (Hirata's synchro-beam)
        delta_star = delta_star + Fz_star+0.5*(\
                    Fx_star*(px_star+0.5*Fx_star)+\
                    Fy_star*(py_star+0.5*Fy_star))
        x_star = x_star - S*Fx_star
        px_star = px_star + Fx_star
        y_star = y_star - S*Fy_star
        py_star = py_star + Fy_star
        
    # Inverse boost on the coordinates of the weak beam
    x_ret, px_ret, y_ret, py_ret, sigma_ret, delta_ret = boost.inv_boost(x_star, px_star, y_star, py_star, sigma_star, delta_star, parboost)

    return x_ret, px_ret, y_ret, py_ret, sigma_ret, delta_ret
