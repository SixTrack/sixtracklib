import numpy as np

# A helper function
def compare(prun, pbench, pbench_prev):
    out = []
    out_rel = []
    error = False
    for att in 'x px y py delta sigma'.split():
        vrun = getattr(prun, att)
        vbench = getattr(pbench, att)
        vbench_prev = getattr(pbench_prev, att)
        diff = vrun-vbench
        diffrel = abs(1.-abs(vrun-vbench_prev)/abs(vbench-vbench_prev))
        out.append(abs(diff))
        out_rel.append(diffrel)
        print(f"{att:<5} {vrun:22.13e} {vbench:22.13e} {diff:22.13g} {diffrel:22.13g}")
        if diffrel > 1e-8 or np.isnan(diffrel):
            if abs(diff) > 1e-11:
                print('Too large discrepancy!')
                error = True
    print(f"\nmax {max(out):21.12e} maxrel {max(out_rel):22.12e}")
    return error


