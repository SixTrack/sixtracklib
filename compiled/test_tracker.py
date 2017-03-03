import numpy as np
import matplotlib.pyplot as plt

from tracker import *

# LATTICE 
lattice = Lattice()

lattice.add("""

{
  LinMap_data data = LinMap_init(0,1,0,1,0,1,0,1, 0.31, 0.32);
  LinMap_track(p,&data);
}

""")


lattice.compile()
#lattice.write_ptx("init.ptx")

lattice.n_turns = 1000
lattice.collect_tbt_data = 1 # every 1 turn

# BUNCH 
bunch = HostBunch(10)
bunch.particle[0].x = 1

# RUN
lattice.track(bunch)

# PLOT
plt.plot(lattice.turns.x(0), lattice.turns.px(0), '.')
plt.show()

