import numpy as np
from cpymad.madx import Madx

class MadPoint(object):
    def __init__(self,name,mad):
        self.name=name
        twiss=mad.table.twiss
        survey=mad.table.survey
        idx=np.where(survey.name==name)[0][0]
        self.tx=twiss.x[idx]
        self.ty=twiss.y[idx]
        self.sx=survey.x[idx]
        self.sy=survey.y[idx]
        self.sz=survey.z[idx]
        theta=survey.theta[idx]
        phi=survey.phi[idx]
        psi=survey.psi[idx]
        thetam=np.array([[np.cos(theta) ,           0,np.sin(theta)],
             [          0,           1,         0],
             [-np.sin(theta),           0,np.cos(theta)]])
        phim=np.array([[          1,          0,          0],
            [          0,np.cos(phi)   ,   np.sin(phi)],
            [          0,-np.sin(phi)  ,   np.cos(phi)]])
        psim=np.array([[   np.cos(psi),  -np.sin(psi),          0],
            [   np.sin(psi),   np.cos(psi),          0],
            [          0,          0,          1]])
        wm=np.dot(thetam,np.dot(phim,psim))
        self.ex=np.dot(wm,np.array([1,0,0]))
        self.ey=np.dot(wm,np.array([0,1,0]))
        self.ez=np.dot(wm,np.array([0,0,1]))
        self.sp=np.array([self.sx,self.sy,self.sz])
        self.p=self.sp+ self.ex * self.tx + self.ey * self.ty
    def dist(self,other):
        return np.sqrt(np.sum((self.p-other.p)**2))
    def distxy(self,other):
        dd=self.p-other.p
        return np.dot(dd,self.ex),np.dot(dd,self.ey)


def add_beambeam(mad,sequence,name,s,from_):
    el=mad.command.beambeam.clone(name)
    mad.seqedit(sequence=sequence)
    mad.install(element=name,at=s,from_=from_)
    mad.endedit()
    return mad.elements[name]

mad=Madx()
mad.options.echo=False;
mad.options.warn=False;
mad.options.info=False;
mad.call("lhc/lhc.seq")
mad.call("lhc/macro.madx")
mad.call("lhc/hllhc_sequence.madx")
mad.exec("mk_beam(7000)")
mad.exec("myslice")
mad.call("lhc/opt_round_150_1500_thin.madx")

mad.globals.on_sep8=2;
mad.use(sequence="lhcb1");
mad.twiss()
mad.survey()
p1=MadPoint('ip8:1',mad)
mad.use(sequence="lhcb2");
mad.twiss()
mad.survey()
p2=MadPoint('ip8:1',mad)

p1.distxy(p2)

add_beambeam(mad,"lhcb1","bb1_b1",15,"ip8")
add_beambeam(mad,"lhcb2","bb2_b2",15,"ip8")



















