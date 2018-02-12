import os

import pyopencl as cl

modulepath=os.path.dirname(os.path.abspath(__file__))
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
srcpath='-I%s'%modulepath
src=open(os.path.join(modulepath,'block.c')).read()
ctx = cl.create_some_context(interactive=False)
prg=cl.Program(ctx,src).build(options=[srcpath])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
rw=mf.READ_WRITE | mf.COPY_HOST_PTR
