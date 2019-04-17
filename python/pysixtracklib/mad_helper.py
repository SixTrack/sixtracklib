from collections import namedtuple


classes = dict(
    Drift=namedtuple('Drift', 'length'),
    Multipole=namedtuple('Multipole', 'knl ksl hxl hyl length'),
    Cavity=namedtuple('Cavity', 'voltage frequency lag'),
    XYShift=namedtuple('XYShift', 'dx dy'),
    SRotation=namedtuple('SRotation', 'angle'),
    Line=namedtuple('Line', 'elems'),
)

def dispatch(el,classes):
    Drift=classes['Drift']
    Multipole=classes['Multipole']
    Cavity=classes['Cavity']

    key=el.base_type.name
    if key=='multipole':
        el=Multipole(knl=el.knl,ksl=el.ksl,hxl=el.knl[0],hyl=0,length=el.lrad)
    elif key in ['marker', 'hmonitor', 'vmonitor', 'instrument',
                 'monitor', 'rcollimator']:
        el=Drift(length=0)
    elif key == 'hkicker':
        el=Multipole(knl=[-el.kick], ksl=[],
                           length=el.lrad, hxl=0, hyl=0)
    elif key == 'vkicker':
        el=Multipole(knl=[], ksl=[el.kick],
                     length=el.lrad, hxl=0, hyl=0)
    elif key == 'rfcavity':
        el=Cavity(voltage=el.volt*1e6,
                  frequency=el.freq*1e6, lag=el.lag*360)
    return el

def madseq_to_line(seq, convert=classes):
    Drift=convert['Drift']
    oldpos = 0
    poss=seq.element_positions()
    names=seq.element_names()
    for name,pos,el in zip(names,poss, seq.elements):
        if el.length > 0:
            raise ValueError(f"Seq {seq} contails {el} with length>0")
        length = pos-oldpos
        if length > 0:
            yield "drift", "Drift", Drift(length)
        el=dispatch(el,convert)
        oldpos=pos
        yield name,el.__class__.__name__,el
