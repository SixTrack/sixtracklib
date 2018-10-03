# tests/sixtracklib/testlib/opencl/kernels/CMakeLists.txt

set( TESTLIB_OPENCL_KERNELS_SOURCES
     opencl_buffer_generic_obj_kernel.cl
     opencl_beam_elements_opencl_kernel.cl
     opencl_particles_kernel.cl
)

# ------------------------------------------------------------------------------
# Install :

if( TESTLIB_OPENCL_KERNELS_SOURCES )
    set( TESTLIB_OPENCL_KERNELS_INSTALL_PATH
         include/sixtracklib/testlib/opencl/kernels )

    install( FILES ${TESTLIB_OPENCL_KERNELS_SOURCES}
             DESTINATION ${TESTLIB_OPENCL_KERNELS_INSTALL_PATH} )

endif()

# end: tests/sixtracklib/testlib/opencl/kernels/CMakeLists.txt
