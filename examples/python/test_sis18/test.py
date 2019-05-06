from cpymad.madx import Madx
import pysixtracklib as pyst

mad = Madx()
mad.options.echo = False

mad.call(file="fodo.madx")
mad.command.beam(particle='proton', energy='6')
mad.use(sequence="FODO")
mad.twiss()

mad.command.select(flag="makethin", class_="quadrupole", slice='8')
mad.command.select(flag="makethin", class_="sbend", slice='8')
mad.command.makethin(makedipedge=False, style="simple", sequence="fodo")

mad.twiss()

sis18 = mad.sequence.FODO

elements = pyst.Elements.from_mad(sis18)
