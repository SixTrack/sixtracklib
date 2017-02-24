# Tracking code exchange format


## Particles

Particle coordinates are given in right handed local reference frame with the axis referred named (`X`,`Y`,`S`). The the largest projection of the momentum is in `s`.

Dimensions are in SI except for energy in GeV and momentum GeV/c.

Coordinates

* id
* turn
* valid
* flag
* s:
* x:
* px:
* y:
* py:
* tau or ptau: at least one between s - beta0 c t, or s/beta0 - ct
* sigma or psigma: canoical
* m0
* P0 or E0 or gamma0 or P0/m0
* m/m0
* q/q0


# Elements

Elements exchanged in the following format:
  `<name type (arg1 ... argn)>`

Elemement lists are exchanged with
  `[ name | (label,name) ] ....

Where the second form use label for lookup and name for physical properties.






