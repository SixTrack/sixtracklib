import numpy as np

from .cobjects import CObject, CProp

class CParticles(CObject):
    clight = 299792458
    pi = 3.141592653589793238
    pcharge = 1.602176565e-19
    echarge = -pcharge
    emass = 0.510998928e6
    pmass = 938.272046e6
    epsilon0 = 8.854187817e-12
    mu0 = 4e-7*pi
    eradius = pcharge**2/(4*pi*epsilon0*emass*clight**2)
    pradius = pcharge**2/(4*pi*epsilon0*pmass*clight**2)
    anumber = 6.022140857e23
    kboltz = 8.6173303e-5  # ev K^-1 #1.38064852e-23 #   JK^-1

    npart   = CProp('u64', 0, 0, const=True)
    charge0 = CProp('f64', 1, 1,     length='npart')
    mass0   = CProp('f64', 2, pmass, length='npart')
    beta0   = CProp('f64', 3, 450e9/np.sqrt(450e9**2+pmass**2), length='npart')
    gamma0  = CProp('f64', 4, np.sqrt(450e9**2+pmass**2)/pmass, length='npart')
    p0c     = CProp('f64', 5, 450e9, length='npart')
    partid  = CProp('u64', 6, 0, length='npart')
    elemid  = CProp('u64', 7, 0, length='npart')
    turn    = CProp('u64', 8, 0, length='npart')
    state   = CProp('u64', 9, 0, length='npart')
    s       = CProp('f64',10, 0, length='npart')
    x       = CProp('f64',11, 0, length='npart')
    px      = CProp('f64',12, 0, length='npart')
    y       = CProp('f64',13, 0, length='npart')
    py      = CProp('f64',14, 0, length='npart')
    sigma   = CProp('f64',15, 0, length='npart')
    psigma  = CProp('f64',16, 0, length='npart')
    delta   = CProp('f64',17, 0, length='npart')
    rpp     = CProp('f64',18, 1, length='npart')
    rvv     = CProp('f64',19, 1, length='npart')
    chi     = CProp('f64',20, 1, length='npart')
    rcharge = CProp('f64',21, 1, length='npart')
    cmass   = CProp('f64',22, 1, length='npart')
    ptau = property(lambda p: (p.psigma*p.beta0))
    pc = property(lambda p: (p.beta*p.gamma*p.mass0))
    energy0 = property(lambda p: (p.gamma0*p.mass0))
    @classmethod
    def from_full_beam(cls, beam):
        npart = len(beam['x'])
        part = cls(npart=npart)
        for nn in part._names:
            setattr(part, nn, getattr(beam, nn))

    def reshape(self, *shape):
        props = self._get_props()
        for offset, name, prop in props:
            if prop.length == 'npart':
                self._shape[name] = shape
        return self

    def compare(self, ref, exclude=['s', 'elemid'], include=[], verbose=True):
        if self.npart == ref.npart:
            names = list(self._names)
            for nn in exclude:
                names.remove(nn)
            for nn in include:
                names.append(nn)
            general = 0
            partn = 1
            fmts = "%-12s: %-14s %-14s %-14s %-14s"
            fmtg = "%-12s:  global diff  %14.6e"
            lgd = ('Variable', 'Reference', 'Value',
                   'Difference', 'Relative Diff')
            lgds = True
            fmt = fmts.replace('-14s', '14.6e')
            for pval, pref in zip(self.particles.flatten(), ref.particles.flatten()):
                pdiff = 0
                for nn in names:
                    val = pval[nn]
                    ref = pref[nn]
                    diff = ref-val
                    if abs(diff) > 0:
                        if abs(ref) > 0:
                            rdiff = diff/ref
                        else:
                            rdiff = diff
                        if lgds:
                            if verbose:
                                print(fmts % lgd)
                                lgds = False
                            if verbose:
                                print(fmt % (nn, ref, val, diff, rdiff))
                        pdiff += rdiff**2
                if pdiff > 0:
                    pl = 'Part %d/%d' % (partn, npart)
                    if verbose:
                        print(fmtg % (pl, np.sqrt(pdiff)))
                    general += pdiff
                partn += 1
            return np.sqrt(general)
        else:
            raise ValueError("Shape ref not compatible")

    def set_delta(self, delta):
        self.delta = delta
        self.rpp = 1./(delta+1)
        pc_eV = self.p0c/self.rpp
        gamma = np.sqrt(1. + (pc_eV/self.mass0)**2)
        beta = np.sqrt(1.-1./gamma**2)
        self.rvv=self.beta0/beta
        self.psigma = self.mass0*(gamma-self.gamma0)/(self.beta0*self.p0c)
