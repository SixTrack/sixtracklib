from cobjects import CBuffer, CObject, CField
import numpy as np


class Particles(CObject):
    pmass = 938.2720813e6

    def _set_p0c(self):
        energy0 = np.sqrt(self.p0c**2 + self.mass0**2)
        self.beta0 = self.p0c / energy0
        self.gamma0 = energy0 / self.mass0

    def _set_delta(self):
        rep = np.sqrt(self.delta**2 + 2 * self.delta + 1 / self.beta0**2)
        irpp = 1 + self.delta
        self.rpp = 1 / irpp
        beta = irpp / rep
        self.rvv = beta / self.beta0
        self.psigma = np.sqrt(self.delta**2 + 2 * self.delta +
                              1 / self.beta0**2) / self.beta0 - 1 / self.beta0**2

    _typeid = 1
    num_particles = CField(0, 'int64', const=True)
    q0 = CField(1, 'real', length='num_particles',
                default=0.0, pointer=True, alignment=8)
    mass0 = CField(2, 'real', length='num_particles',
                   default=pmass, pointer=True, alignment=8)
    beta0 = CField(3, 'real', length='num_particles',
                   default=1.0, pointer=True, alignment=8)
    gamma0 = CField(4, 'real', length='num_particles',
                    default=1.0, pointer=True, alignment=8)
    p0c = CField(5, 'real', length='num_particles',
                 default=1.0, pointer=True, alignment=8, setter=_set_p0c)
    s = CField(6, 'real', length='num_particles',
               default=0.0, pointer=True, alignment=8)
    x = CField(7, 'real', length='num_particles',
               default=0.0, pointer=True, alignment=8)
    y = CField(8, 'real', length='num_particles',
               default=0.0, pointer=True, alignment=8)
    px = CField(9, 'real', length='num_particles',
                default=0.0, pointer=True, alignment=8)
    py = CField(10, 'real', length='num_particles',
                default=0.0, pointer=True, alignment=8)
    zeta = CField(11, 'real', length='num_particles',
                  default=0.0, pointer=True, alignment=8)
    psigma = CField(12, 'real', length='num_particles',
                    default=0.0, pointer=True, alignment=8)
    delta = CField(13, 'real', length='num_particles',
                   default=0.0, pointer=True, alignment=8, setter=_set_delta)
    rpp = CField(14, 'real', length='num_particles',
                 default=1.0, pointer=True, alignment=8)
    rvv = CField(15, 'real', length='num_particles',
                 default=1.0, pointer=True, alignment=8)
    chi = CField(16, 'real', length='num_particles',
                 default=1.0, pointer=True, alignment=8)
    charge_ratio = CField(17, 'real', length='num_particles',
                          default=1.0, pointer=True, alignment=8)
    particle_id = CField(18, 'int64', length='num_particles',
                         default=0, pointer=True, alignment=8)
    at_element = CField(19, 'int64', length='num_particles',
                        default=0, pointer=True, alignment=8)
    at_turn = CField(20, 'int64', length='num_particles',
                     default=0, pointer=True, alignment=8)
    state = CField(21, 'int64', length='num_particles',
                   default=1, pointer=True, alignment=8)

    @classmethod
    def from_ref(cls, num_particles=1, mass0=pmass,
                 p0c=1e9, q0=1):
        return cls(num_particles=num_particles,
                   particle_id=np.arange(num_particles),
                   ).set_reference(mass0=mass0, p0c=p0c, q0=q0)

    sigma = property(lambda self: (self.beta0 / self.beta) * self.zeta)
    beta = property(lambda p: (1 + p.delta) / (1 / p.beta0 + p.ptau))

    @property
    def ptau(self):
        return np.sqrt(self.delta**2 + 2 * self.delta +
                       1 / self.beta0**2) - 1 / self.beta0

    def set_reference(self, p0c=7e12, mass0=pmass, q0=1):
        self.q0 = q0
        self.mass0 = mass0
        self.p0c = p0c
        return self

    def from_pysixtrack(self, inp, particle_index):
        assert(particle_index < self.num_particles)
        self.q0[particle_index] = inp.q0
        self.mass0[particle_index] = inp.mass0
        self.beta0[particle_index] = inp.beta0
        self.gamma0[particle_index] = inp.gamma0
        self.p0c[particle_index] = inp.p0c
        self.s[particle_index] = inp.s
        self.x[particle_index] = inp.x
        self.y[particle_index] = inp.y
        self.px[particle_index] = inp.px
        self.py[particle_index] = inp.py
        self.zeta[particle_index] = inp.zeta
        self.psigma[particle_index] = inp.psigma
        self.delta[particle_index] = inp.delta
        self.rpp[particle_index] = inp.rpp
        self.rvv[particle_index] = inp.rvv
        self.chi[particle_index] = inp.chi
        self.charge_ratio[particle_index] = inp.qratio
        self.particle_id[particle_index] = \
            inp.partid is not None and inp.partid or particle_index
        self.at_element[particle_index] = inp.elemid
        self.at_turn[particle_index] = inp.turn
        self.state[particle_index] = inp.state
        return

    def to_pysixtrack(self, other, particle_index):
        assert(particle_index < self.num_particles)
        other._update_coordinates = False
        other.q0 = self.q0[particle_index]
        other.mass0 = self.mass0[particle_index]
        other.beta0 = self.beta0[particle_index]
        other.gamma0 = self.gamma0[particle_index]
        other.p0c = self.p0c[particle_index]
        other.s = self.s[particle_index]
        other.x = self.x[particle_index]
        other.y = self.y[particle_index]
        other.px = self.px[particle_index]
        other.py = self.py[particle_index]
        other.zeta = self.zeta[particle_index]
        other.psigma = self.psigma[particle_index]
        other.delta = self.delta[particle_index]
        other.chi = self.chi[particle_index]
        other.qratio = self.charge_ratio[particle_index]
        other.partid = self.particle_id[particle_index]
        other.turn = self.at_turn[particle_index]
        other.elemid = self.at_element[particle_index]
        other.state = self.state[particle_index]
        other._update_coordinates = True

        return


def makeCopy(orig, cbuffer=None):
    p = Particles(
        cbuffer=cbuffer,
        num_particles=orig.num_particles,
        q0=orig.q0,
        mass0=orig.mass0,
        beta0=orig.beta0,
        gamma0=orig.gamma0,
        p0c=orig.p0c,
        s=orig.s,
        x=orig.x,
        y=orig.y,
        px=orig.px,
        py=orig.py,
        zeta=orig.zeta,
        delta=orig.delta,
        psigma=orig.psigma,
        rpp=orig.rpp,
        rvv=orig.rvv,
        chi=orig.chi,
        charge_ratio=orig.charge_ratio,
        particle_id=orig.particle_id,
        at_element=orig.at_element,
        at_turn=orig.at_turn,
        state=orig.state)

    return p


def calcParticlesDifference(lhs, rhs, cbuffer=None):
    assert(lhs.num_particles == rhs.num_particles)
    diff = Particles(num_particles=lhs.num_particles, cbuffer=cbuffer)
    keys = ['q0', 'mass0', 'beta0', 'gamma0', 'p0c', 's', 'x', 'y', 'px', 'py',
            'zeta', 'psigma', 'delta', 'rpp', 'rvv', 'chi', 'charge_ratio',
            'particle_id', 'at_element', 'at_turn', 'state']

    for k in keys:
        try:
            setattr(diff, k, getattr(lhs, k) - getattr(rhs, k))
        except TypeError as err:
            pass

    return diff


def compareParticlesDifference(lhs, rhs, abs_treshold=None):
    cmp_result = -1

    if lhs.num_particles == rhs.num_particles and lhs.num_particles > 0:
        num_particles = lhs.num_particles

        real_keys = [
            'q0', 'mass0', 'beta0', 'gamma0', 'p0c', 's', 'x', 'y', 'px', 'py',
            'zeta', 'psigma', 'delta', 'rpp', 'rvv', 'chi', 'charge_ratio']

        cmp_result = 0

        for k in real_keys:
            lhs_arg = getattr(lhs, k)
            rhs_arg = getattr(rhs, k)

            if np.array_equal(lhs_arg, rhs_arg):
                continue

            if abs_treshold is not None and abs_treshold > 0.0:
                diff = np.absolute(lhs_arg - rhs_arg)
                for ii in range(0, num_particles):
                    if diff[ii] > abs_treshold:
                        cmp_result = lhs_arg[ii] > rhs_arg[ii] and +1 or -1
                        break
            else:
                for ii in range(0, num_particles):
                    if lhs_arg[ii] > rhs_arg[ii]:
                        cmp_result = +1
                        break
                    elif lhs_arg[ii] < rhs_arg[ii]:
                        cmp_result = -1
                        break

            if cmp_result != 0:
                break

        if cmp_result == 0:
            int_keys = ['particle_id', 'at_element', 'at_turn', 'state']

            for k in int_keys:
                lhs_arg = getattr(lhs, k)
                rhs_arg = getattr(rhs, k)

                if np.array_equal(lhs_arg, rhs_arg):
                    continue

                for ii in range(0, num_particles):
                    if lhs_arg[ii] > rhs_arg[ii]:
                        cmp_result = +1
                        break
                    elif lhs_arg[ii] < rhs_arg[ii]:
                        cmp_result = -1
                        break

                if cmp_result != 0:
                    break

    return cmp_result


class ParticlesSet(object):
    element_types = {'Particles': Particles}

    @property
    def particles(self):
        return self.cbuffer.get_objects()

    def __init__(self, cbuffer=None):

        if cbuffer is None:
            self.cbuffer = CBuffer()
        else:
            self.cbuffer = cbuffer
        for name, cls in self.element_types.items():
            self.cbuffer.typeids[cls._typeid] = cls

    def Particles(self, **nargs):
        particles = Particles(cbuffer=self.cbuffer, **nargs)
        return particles

    @classmethod
    def fromfile(cls, filename):
        cbuffer = CBuffer.fromfile(filename)
        return cls(cbuffer=cbuffer)

    @classmethod
    def fromSixDump101(cls, input_folder, st_dump_file, **kwargs):

        import sixtracktools
        import pysixtrack
        six = sixtracktools.SixInput(input_folder)
        line, rest, iconv = six.expand_struct(convert=pysixtrack.element_types)

        sixdump = sixtracktools.SixDump101(st_dump_file)

        num_iconv = int(len(iconv))
        num_belem = int(len(line))
        num_dumps = int(len(sixdump.particles))

        assert(num_iconv > 0)
        assert(num_belem > iconv[num_iconv - 1])
        assert(num_dumps >= num_iconv)
        assert((num_dumps % num_iconv) == 0)

        num_particles = int(num_dumps / num_iconv)

        self = cls(**kwargs)

        for ii in range(num_iconv):
            elem_id = iconv[ii]
            assert(elem_id < num_belem)

            p = self.Particles(num_particles=num_particles)

            assert(p.num_particles == num_particles)
            assert(len(p.q0) == num_particles)

            for jj in range(num_particles):
                kk = num_particles * ii + jj
                assert(kk < num_dumps)
                p.from_pysixtrack(
                    pysixtrack.Particles(**sixdump[kk].get_minimal_beam()), jj)
                p.state[jj] = 1
                p.at_element[jj] = elem_id

        return self

    def to_file(self, filename):
        self.cbuffer.tofile(filename)
