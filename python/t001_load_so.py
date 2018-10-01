import os, ctypes
libpath='/home/giadarol/Desktop/20180925_sixtracklib/sixtracklib/build/sixtracklib/libsixtrack.so'
slib=ctypes.CDLL(libpath)

libpath='/home/giadarol/Desktop/20180925_sixtracklib/sixtracklib/build/tests/sixtracklib/testlib/libsixtrack_test.so'
tslib=ctypes.CDLL(libpath)


# Particle buffer
part_buf = slib.st_Buffer_new_from_file(ctypes.create_string_buffer(b"bbsimple_st_dump.bin"))

# Elem buffer
ele_buf = slib.st_Buffer_new_from_file(ctypes.create_string_buffer(b"bbsimple_st_input.bin"))

