# Tracking Data Exchange Conventions

## Particles

Particle coordinates are given in right handed local reference frame with the axis referred named (`X`,`Y`,`S`) where the largest projection of the momentum is in `S`. Dimensions are in SI except for energy in GeV and momentum GeV/c.

Coordinates quantities

* `partid`: particle id
* `state`: >=0 particle valid, <0 particle frozen
* `elemid`: elem where the particle si
* `turn`  : turn number
* `s`: path length in the reference trajectory
* `x`: transverse plane 
* `px`: transverse normalized momentum Px/P0
* `xp`: transverse divergence Px/P
* `y` : transverse normalized momentum Px/P0
* `py`: transverse divergence Px/P
* `tau`: delay : s/beta0 - c T
* `ptau`: normalized energy deviation: (energy-energy0)/p0c
* `sigma`: delay : s - beta0 c T
* `psigma`: normalized energy deviation: (energy-energy0)/(\beta0 p0c)
* `delta`: relative momentum deviation: (pc -p0c)/p0c
* `rpp`: beta0/beta
* `rvv`: p0c/pc = 1/(1+delta)
* `energy`: 
* `pc`: 
* `beta`: pc/energy
* `gamma`: energy/mass
* `mass`: 
* `charge`: 
* `mratio`: 
* `qratio`: 
* `chi`: 
* `mass0`  [eV/c^2]:  reference mass
* `energy0`[eV]: reference energy 
* `p0c`    [eV]:
* `gamma0` [1]:
* `beta0`  [1]:
* `charge0`[1]: number of e charges





Particles data is exchanged a dictionary where quantities can be scalars or vectors of the same shape

# Elements

Elements exchanged in the following format:
  `<name type arguments>`

Elemement lists are exchanged with
  `[ name | (label,name) ] ....`

where the second form uses label for lookup and name for physical properties.


Elements types are:

* drift: l[m]
* driftexact: l[m]
* multipole: knl[m^-n] ksl[m^-n] l hxl hyl
* cavity: volt[V] freq[Hz] lag[degree]
* align: dx[m] dy[m] tilt[degree]

## Align

Shift by dx dy tilt the reference frame (e.g. apply negative signs to the variables)

* dx [m]
* dy [m]
* tilt [m]


