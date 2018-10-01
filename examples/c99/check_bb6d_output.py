import cobjects

import temp_particles as tp

dump_to_be_loaded = "../../build/examples/c99/stlib_dump.bin"
buf = cobjects.CBuffer.from_file(dump_to_be_loaded)

particles = []
for iob in range(buf.n_objects):
    particles.append(buf.get_object(tp.Particles, iob))


