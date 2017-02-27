import numpy as np

def twiss2matrix(alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1, D_x_s1,
                 alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1, D_y_s1,
                 dQ_x, dQ_y):
  I = np.zeros((4, 4))
  J = np.zeros((4, 4))

  # Sine component.
  I[0,0] = np.sqrt(beta_x_s1 / beta_x_s0)
  I[0,1] = 0.
  I[1,0] = (np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
           (alpha_x_s0 - alpha_x_s1))
  I[1,1] = np.sqrt(beta_x_s0 / beta_x_s1)
  I[2,2] = np.sqrt(beta_y_s1 / beta_y_s0)
  I[2,3] = 0.
  I[3,2] = (np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
           (alpha_y_s0 - alpha_y_s1))
  I[3,3] = np.sqrt(beta_y_s0 / beta_y_s1)

  # Cosine component.
  J[0,0] = np.sqrt(beta_x_s1 / beta_x_s0) * alpha_x_s0
  J[0,1] = np.sqrt(beta_x_s0 * beta_x_s1)
  J[1,0] = -(np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
           (1. + alpha_x_s0 * alpha_x_s1))
  J[1,1] = -np.sqrt(beta_x_s0 / beta_x_s1) * alpha_x_s1
  J[2,2] = np.sqrt(beta_y_s1 / beta_y_s0) * alpha_y_s0
  J[2,3] = np.sqrt(beta_y_s0 * beta_y_s1)
  J[3,2] = -(np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
           (1. + alpha_y_s0 * alpha_y_s1))
  J[3,3] = -np.sqrt(beta_y_s0 / beta_y_s1) * alpha_y_s1

  dphi_x = 2.*np.pi*dQ_x
  dphi_y = 2.*np.pi*dQ_y

  c_dphi_x = np.cos(dphi_x)
  c_dphi_y = np.cos(dphi_y)
  s_dphi_x = np.sin(dphi_x)
  s_dphi_y = np.sin(dphi_y)

  M00 = I[0,0] * c_dphi_x + J[0,0] * s_dphi_x
  M01 = I[0,1] * c_dphi_x + J[0,1] * s_dphi_x
  M10 = I[1,0] * c_dphi_x + J[1,0] * s_dphi_x
  M11 = I[1,1] * c_dphi_x + J[1,1] * s_dphi_x
  M22 = I[2,2] * c_dphi_y + J[2,2] * s_dphi_y
  M23 = I[2,3] * c_dphi_y + J[2,3] * s_dphi_y
  M32 = I[3,2] * c_dphi_y + J[3,2] * s_dphi_y
  M33 = I[3,3] * c_dphi_y + J[3,3] * s_dphi_y

  return np.array([M00, M01, M10, M11, M22, M23, M32, M33])
