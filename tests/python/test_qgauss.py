import numpy as np
import sixtracklib as st
from sixtracklib.stcommon import Math_q_gauss, Math_sqrt_beta_from_gauss_sigma

def test_qgauss_gauss_compare():
    def gauss_distr(x, sigma, mu=0.0):
        sigma_squ = sigma * sigma
        arg = (x - mu) / sigma
        return np.exp(-arg * arg / 2.0) / np.sqrt(2 * np.pi * sigma_squ)

    q = 1.0
    sigma1 = 1.0
    abscissa = np.linspace(-4 * sigma1, 4 * sigma1, 101)
    cmp_gauss1 = gauss_distr(abscissa, sigma1)

    EPS = float(1e-16)
    sqrt_beta1 = Math_sqrt_beta_from_gauss_sigma( sigma1 )
    gauss1 = np.array( [ Math_q_gauss( x, q, sqrt_beta1 ) for x in abscissa ] )
    assert np.allclose(cmp_gauss1, gauss1, EPS, EPS)

    sigma2 = 2.37
    sqrt_beta2 = Math_sqrt_beta_from_gauss_sigma( sigma2 )
    abscissa = np.linspace(-4 * sigma2, 4 * sigma2, 101)
    cmp_gauss2 = gauss_distr( abscissa, sigma2 )
    gauss2 = np.array( [ Math_q_gauss( x, q, sqrt_beta2 ) for x in abscissa ] )
    assert np.allclose(cmp_gauss2, gauss2, EPS, EPS)

if __name__ == "__main__":
    test_qgauss_gauss_compare()
