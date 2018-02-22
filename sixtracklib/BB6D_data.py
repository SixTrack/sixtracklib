import propagate_sigma_matrix as psm
import boost as boost
import slicing

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
    
