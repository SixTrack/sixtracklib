# ==============================================================================
# tests/sixtracklib/cuda/wrappers/CMakeLists.txt

set( SIXTRL_TEST_LIBS ${SIXTRACKL_TEST_LIBRARIES}
     ${SIXTRACKL_GTEST_LIBRARIES} sixtrack_test sixtrack )

if( GTEST_FOUND )
    set( UNIT_TEST_TARGETS )

    # ==========================================================================
    # test_particles_store_all_addresses_cuda_c99:

    add_executable( test_particles_store_all_addresses_cuda_c99
        test_particles_store_all_addresses_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
        test_particles_store_all_addresses_cuda_c99 )

    target_link_libraries( test_particles_store_all_addresses_cuda_c99
        ${SIXTRL_TEST_LIBS} )

    add_test( C99_CudaWrappersParticlesStoreAddrTests,
              test_particles_store_all_addresses_cuda_c99 )

    # --------------------------------------------------------------------------
    # test_track_job_cuda_c99:

#     add_executable( test_track_job_cuda_c99 test_track_job_c99.cpp )
#     set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_cuda_c99 )
#     target_link_libraries( test_track_job_cuda_c99 ${SIXTRL_TEST_LIBS} )
#     add_test( CXX_CudaControllerTests test_track_job_cuda_c99 )

endif()

if( UNIT_TEST_TARGETS )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY POSITION_INDEPENDENT_CODE ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY LINKER_LANGUAGE C )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_SEPARABLE_COMPILATION  ON  )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_STANDARD_REQUIRED 11 )
endif()

# end: tests/sixtracklib/cuda/control/CMakeLists.txt
