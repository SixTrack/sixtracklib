import numpy as np
import sixtracklib


block=sixtracklib.cBlock(2)
block.Multipole([1.,3.,5.],[2.,4.,6.],0,0,0,)
block.Drift(56.)
block.Drift(5.)
block.Block()

data = block.data['u64'][0:block.last]
print("".join(["%lu, " % (i, ) for i in data]))
