import numpy as np

def track_mad_linmap_and_beambeam(intensity_pbun, energy_GeV, nemittx, nemitty, tune_x, tune_y, 
      beta_s, include_beambeam, offsetx_s, offsety_s, sigmax_s, sigmay_s, x0_particles, px0_particles, y0_particles, py0_particles, nturns):

    assert(len(x0_particles) == len(px0_particles) == len(y0_particles) == len(py0_particles))
    
    x_particles = []
    y_particles = []
    px_particles = []
    py_particles = []
    
    for i_part, (x_part, px_part, y_part, py_part) in enumerate(zip(x0_particles, px0_particles, y0_particles, py0_particles)):
        #~ print i_part
        with open('mad_auto.madx', 'w') as fmad:
            fmad.write("beam, particle=proton, npart = %.2fe11, energy=%.2f, exn=%e, eyn=%e;\n\n"%(intensity_pbun/1e11, energy_GeV, nemittx, nemitty))
            fmad.write("tune_x = %.4f;\ntune_y =  %.4f;\nbeta_s = %.4f;\n"%(tune_x, tune_y, beta_s))

            fmad.write("""
            one_turn: matrix, 
            rm11= cos(2*pi*tune_x),        rm12=sin(2*pi*tune_x)*beta_s,
            rm21=-sin(2*pi*tune_x)/beta_s, rm22=cos(2*pi*tune_x),
            rm33= cos(2*pi*tune_y),        rm34=sin(2*pi*tune_y)*beta_s,
            rm43=-sin(2*pi*tune_y)/beta_s, rm44=cos(2*pi*tune_y)  

            ;

            linmap:   line=(one_turn);

            """)
            myring_string = 'linmap'


            # Insert beam beam
            if include_beambeam:
              fmad.write("beam_beam: beambeam, charge=1, xma=%e, yma=%e, sigx=%e, sigy=%e;"%(offsetx_s, offsety_s, sigmax_s, sigmay_s))
              myring_string += ', beam_beam'



            # Finalize and track
            fmad.write("""
            myring: line=(%s);
            use,period=myring;


            """%myring_string)

            #Track
            fmad.write("track, dump;\n")
            fmad.write("start, x= %e, px=%e, y=%e, py=%e;\n"%(x_part, px_part, y_part, py_part))

            fmad.write("run,turns=%d;\nendtrack;\n"%nturns)

        mad_executable = 'madx'

        import os
        try:
            os.remove('track.obs0001.p0001')
        except OSError as err:
            print err

        import subprocess as sp
        sp.call((mad_executable,  'mad_auto.madx'))

        print 'Part %d/%d'%(i_part, len(x0_particles))

        import metaclass 
        ob = metaclass.twiss('track.obs0001.p0001')
        
        x_particles.append(ob.X.copy())
        y_particles.append(ob.Y.copy())
        
        px_particles.append(ob.PX.copy())
        py_particles.append(ob.PY.copy())
        
    return np.array(x_particles), np.array(y_particles), np.array(px_particles), np.array(py_particles)
        
        
    
