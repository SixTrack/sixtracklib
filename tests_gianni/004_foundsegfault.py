import sys; sys.path.append('../')

import sixtracklib
machine = sixtracklib.CBlock()
machine.add_Drift(length=1.)

print('This works')
bunch=sixtracklib.CParticles(npart=1)
particles,ebe,tbt=machine.track_cl(bunch,nturns=1,elembyelem=True,turnbyturn=True)
bunch=sixtracklib.CParticles(npart=1)
particles,ebe,tbt=machine.track_cl(bunch,nturns=1,elembyelem=True,turnbyturn=True)
print('Done')

# print('This gives segfault:')
# bunch=sixtracklib.CParticles(npart=1)
# particles,ebe,tbt=machine.track_cl(bunch,nturns=1,elembyelem=True,turnbyturn=True)
# particles,ebe,tbt=machine.track_cl(bunch,nturns=1,elembyelem=True,turnbyturn=True)

print('This also gives segfault:')
bunch=sixtracklib.CParticles(npart=1)
particles,ebe,tbt=machine.track_cl(bunch,nturns=1,elembyelem=True,turnbyturn=True)
particles2,ebe,tbt=machine.track_cl(particles,nturns=1,elembyelem=True,turnbyturn=True)