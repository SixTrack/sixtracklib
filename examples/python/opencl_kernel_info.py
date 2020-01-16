import sixtracklib as st
from sixtracklib.stcommon import st_ClContext_uses_optimized_tracking
from sixtracklib.stcommon import st_ClContextBase_is_debug_mode_enabled
from sixtracklib.stcommon import st_ARCH_ILLEGAL_KERNEL_ID

if __name__ == '__main__':
    lattice = st.Elements()
    drift = lattice.Drift(length=1.0)

    pset = st.Particles(num_particles=1000)

    job = st.TrackJob(lattice, pset, device="opencl:0.0")
    ctrl = job.controller

    assert not st_ClContextBase_is_debug_mode_enabled( ctrl.pointer )
    assert st_ClContext_uses_optimized_tracking( ctrl.pointer )

    k_id = ctrl.find_kernel_by_name(
        "st_Track_particles_until_turn_opt_pp_opencl" )

    assert k_id != st_ARCH_ILLEGAL_KERNEL_ID.value
    print( f""""
          current workgroup size (0 == max) : {ctrl.kernel_workgroup_size( k_id )}
          max workgroup size                : {ctrl.kernel_max_workgroup_size(k_id )}
          preferred workgroup size multiple : {ctrl.kernel_preferred_workgroup_size_multiple(k_id)}
          """ )
