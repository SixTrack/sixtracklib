import os

from pyoptics import madlang, MadBeam

import sixtracklib

def newer(f1,f2):
    return os.path.getmtime(f1)>os.path.getmtime(f2)

mad=madlang.open('bench.madx')
line,rest=mad.bench.expand_struct()
if newer('bench.madx','track.obs0001.p0001'):
  os.system('madx bench.madx')

madbeam=MadBeam.open('track.obs0001.p0001','twiss_bench.tfs')

block=sixtracklib.cBlock.from_line(line)
bref=sixtracklib.cBeam.from_full_beam(madbeam.get_full_beam())
cbeam=bref.copy()[[0]]

block.track(cbeam,nturn=2,turnbyturn=True)

bnew=block.turnbyturn[:,0]

bnew.compare(bref)
