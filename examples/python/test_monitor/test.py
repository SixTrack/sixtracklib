import sixtracklib
sixtracklib.TrackJobCL.print_nodes()

part = sixtracklib.Particles(
    num_particles=16).set_reference(
        p0c=7e12, mass0=938.272046e6, q0=1)

elem = sixtracklib.Elements()
elem.BeamMonitor(num_stores=10)
elem.Drift(length=5.0)
elem.Multipole(knl=[0.1570796327], ksl=[], hxl=0.1570796327, hyl=0, length=0.0)
elem.Drift(length=5.0)
elem.Multipole(knl=[0, 0.1657145946], ksl=[0, 0], hxl=0, hyl=0, length=0)
elem.Drift(length=5.0)
elem.Multipole(knl=[0.1570796327], ksl=[], hxl=0.1570796327, hyl=0, length=0.0)
elem.Drift(length=5.0)
elem.Multipole(knl=[0, -0.1685973315], ksl=[0, 0], hxl=0, hyl=0, length=0)
elem.Cavity(voltage=5000000.0, frequency=239833966.4, lag=180)


job = sixtracklib.TrackJobCL(part, elem, "0.0")
job.track_until(10)
