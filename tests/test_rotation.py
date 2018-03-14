import numpy as np
import matplotlib.pyplot as plt

import sys; sys.path.append('../')

import sixtracklib

def test_track():
  fRF      = 400e6
  V_RF     = 16e6
  lag_deg  = 180
  p0c_eV   = 6.5e12  
  qx       = 0.31
  qy       = 0.32
  betax    = 122.21
  betay    = 210.1048
  h        = 35640.
  #alfax    = 2.374318411
  #alfay    = -2.485300082
  alfax   = 0.0
  alfay   = 0.
  gamma_tr = 55.68
  pmass_eV = 938.272046e6
  exn      = 2.5e-6  
  eyn      = 2.5e-6  
  npart    = 10**2
  nturns   = 2*556
  
  ap = 1./(gamma_tr**2)

  gamma0   = np.sqrt(p0c_eV**2+pmass_eV**2)/pmass_eV
  beta0    = p0c_eV/np.sqrt(p0c_eV**2+pmass_eV**2)
  egeomx   = exn/(beta0*gamma0)
  egeomy   = eyn/(beta0*gamma0)
  sigma_x  = np.sqrt(egeomx*betax)
  sigma_px = np.sqrt(egeomx/betax)
  sigma_y  = np.sqrt(egeomy*betay)
  sigma_py = np.sqrt(egeomy/betay)

  machine=sixtracklib.CBlock()
  machine.add_LinearMap(qx=qx,qy=qy,betax=betax,betay=betay,alfax=alfax,alfay=alfay,ap=ap,h=h, fRF=fRF)
  machine.add_Cavity(voltage=V_RF,frequency=fRF,lag=lag_deg)

  bunch=sixtracklib.CParticles(npart=npart,
                p0c=p0c_eV,
                beta0 = beta0,
                gamma0 = gamma0)
 
  
  J_min  = 1e-11
  Jx_max = 1e-9
  Jy_max = 1e-9
  Jx = np.linspace(J_min,Jx_max,np.sqrt(npart))
  Jy = np.linspace(J_min,Jy_max,np.sqrt(npart))
  x  = np.sqrt(Jx*betax*2)
  y  = np.sqrt(Jy*betay*2)
  xx, yy  = np.meshgrid(x,y)
  bunch.x = xx.flatten()
  bunch.y = yy.flatten()
  
  #bunch.px = np.linspace(0,1e-3,npart)
  #bunch.py = np.linspace(0,1e-3,npart)
  bunch.set_delta(np.linspace(0,3.5e-4,npart))
  

  fig1,ax1 = plt.subplots()
  plt.plot(bunch.x/sigma_x, bunch.y/sigma_y, c='k', marker='o' , ms=2,linestyle= ' ')
  ax1.set_xlabel(r"$ \rm x[\sigma]$",fontsize=14)
  ax1.set_ylabel(r"$ \rm y[\sigma]$",fontsize=14)
  ax1.grid()
  fig1.tight_layout()

  particles,ebe,tbt=machine.track_cl(bunch,nturns=nturns,
                                  elembyelem=None,turnbyturn=True)

  fig2,ax2 = plt.subplots()
  fig3,ax3 = plt.subplots()
  jet= plt.get_cmap('jet')
  colors = iter(jet(np.linspace(0,1,npart)))
  for i in range (npart):  
    c1 = next(colors)
    ax2.plot(tbt.y[:,i], tbt.py[:,i], c=c1, ms=2,marker='o',linestyle=' ')
    ax3.plot(tbt.sigma[:,i], tbt.psigma[:,i], c=c1, ms=2,marker='o',linestyle=' ')
  ax2.set_xlabel('y',fontsize=14)
  ax2.set_ylabel(r'$ \rm p_y$',fontsize=14)
  ax3.set_xlabel('sigma',fontsize=14)
  ax3.set_ylabel('psigma',fontsize=14)
  fig2.tight_layout()
  fig3.tight_layout()

  fig4,ax4 = plt.subplots(nrows=2,figsize=(9,6))
  for i in range(npart):
    fourier = np.fft.fft(tbt.x[:,i] + 1j*tbt.px[:,1])
    freq = np.fft.fftfreq(len(fourier))
    ax4[0].plot(freq, abs(fourier)/max(abs(fourier)))
    fourier = np.fft.fft(tbt.y[:,i] + 1j*tbt.py[:,1])
    freq = np.fft.fftfreq(len(fourier))
    ax4[1].plot(freq, abs(fourier)/max(abs(fourier)))
  ax4[0].set_xlim([0.2,0.4])
  ax4[0].set_xlabel('Bin',fontsize=14)
  ax4[0].set_ylabel(r'x-plane',fontsize=14)
  ax4[0].grid()
  ax4[1].set_xlim([0.2,0.4])
  ax4[1].set_xlabel('Bin',fontsize=14)
  ax4[1].set_ylabel(r'y-plane',fontsize=14)
  ax4[1].grid()
  fig4.tight_layout()

  plt.show()
  return machine,particles,ebe,tbt


if __name__=='__main__':
    machine,particles,ebe,tbt=test_track()


