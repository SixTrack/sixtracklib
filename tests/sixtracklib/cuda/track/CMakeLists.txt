# ==============================================================================
# tests/sixtracklib/cuda/track/CMakeLists.txt

set( SIXTRL_TEST_LIBS ${SIXTRACKL_TEST_LIBRARIES}
     ${SIXTRACKL_GTEST_LIBRARIES} sixtrack_test sixtrack )

if( GTEST_FOUND )
    set( UNIT_TEST_TARGETS )

    # ==========================================================================
    # test_track_job_setup_cuda_cxx:

    add_executable( test_track_job_setup_cuda_cxx test_track_job_setup_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_setup_cuda_cxx )
    target_link_libraries( test_track_job_setup_cuda_cxx ${SIXTRL_TEST_LIBS} )
    add_test( CXX_CudaTrackJobSetupTests test_track_job_setup_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_setup_cuda_c99:

    add_executable( test_track_job_setup_cuda_c99 test_track_job_setup_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_setup_cuda_c99 )
    target_link_libraries( test_track_job_setup_cuda_c99 ${SIXTRL_TEST_LIBS} )
    add_test( C99_CudaTrackJobSetupTests test_track_job_setup_cuda_c99 )

    # ==========================================================================
    # test_track_job_collect_push_cuda_cxx:

    add_executable( test_track_job_collect_push_cuda_cxx
                    test_track_job_collect_push_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_collect_push_cuda_cxx )

    target_link_libraries( test_track_job_collect_push_cuda_cxx
                           ${SIXTRL_TEST_LIBS} )

    add_test( C99_CudaTrackJobCollectAndPushTests
              test_track_job_collect_push_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_collect_push_cuda_c99:

    add_executable( test_track_job_collect_push_cuda_c99
                    test_track_job_collect_push_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_collect_push_cuda_c99 )

    target_link_libraries( test_track_job_collect_push_cuda_c99
                           ${SIXTRL_TEST_LIBS} )

    add_test( C99_CudaTrackJobCollectAndPushTests
              test_track_job_collect_push_cuda_c99 )

    # ==========================================================================
    # test_track_job_fetch_particle_addr_cuda_cxx:

    add_executable( test_track_job_fetch_particle_addr_cuda_cxx
                    test_track_job_fetch_particle_addr_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_fetch_particle_addr_cuda_cxx )

    target_link_libraries( test_track_job_fetch_particle_addr_cuda_cxx
                          ${SIXTRL_TEST_LIBS} )

    add_test( CXX_CudaTrackJobFetchParticleAddrTests
              test_track_job_fetch_particle_addr_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_fetch_particle_addr_cuda_c99:

    add_executable( test_track_job_fetch_particle_addr_cuda_c99
                    test_track_job_fetch_particle_addr_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_fetch_particle_addr_cuda_c99 )

    target_link_libraries( test_track_job_fetch_particle_addr_cuda_c99
                          ${SIXTRL_TEST_LIBS} )

    add_test( C99_CudaTrackJobFetchParticleAddrTests
              test_track_job_fetch_particle_addr_cuda_c99 )

    # ==========================================================================
    # test_track_job_track_until_cuda_cxx:

    add_executable( test_track_job_track_until_cuda_cxx
                    test_track_job_track_until_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_until_cuda_cxx )

    target_link_libraries( test_track_job_track_until_cuda_cxx
                           ${SIXTRL_TEST_LIBS} )

    add_test( CXX_CudaTrackJobTrackUntilTests
              test_track_job_track_until_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_track_until_cuda_c99:

    add_executable( test_track_job_track_until_cuda_c99
                    test_track_job_track_until_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_until_cuda_c99 )

    target_link_libraries( test_track_job_track_until_cuda_c99
                           ${SIXTRL_TEST_LIBS} )

    add_test( C99_CudaTrackJobTrackUntilTests
              test_track_job_track_until_cuda_c99 )

    # ==========================================================================
    # test_track_job_track_line_cuda_cxx:

    add_executable( test_track_job_track_line_cuda_cxx
                    test_track_job_track_line_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_line_cuda_cxx )

    target_link_libraries( test_track_job_track_line_cuda_cxx
                           ${SIXTRL_TEST_LIBS} )

    add_test( CXX_CudaTrackJobTrackLineTests
              test_track_job_track_line_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_track_line_cuda_c99:

    add_executable( test_track_job_track_line_cuda_c99
                    test_track_job_track_line_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_line_cuda_c99 )

    target_link_libraries( test_track_job_track_line_cuda_c99
                           ${SIXTRL_TEST_LIBS} )

    add_test( C99_CudaTrackJobTrackLineTests
              test_track_job_track_line_cuda_c99 )

    # ==========================================================================
    # test_track_job_track_elem_by_elem_cuda_cxx:

    add_executable( test_track_job_track_elem_by_elem_cuda_cxx
                    test_track_job_track_elem_by_elem_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_elem_by_elem_cuda_cxx )

    target_link_libraries( test_track_job_track_elem_by_elem_cuda_cxx
                          ${SIXTRL_TEST_LIBS} )

    add_test( CXX_CudaTrackJobTrackElemByElemTests
              test_track_job_track_elem_by_elem_cuda_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_track_elem_by_elem_cuda_c99:

    add_executable( test_track_job_track_elem_by_elem_cuda_c99
                    test_track_job_track_elem_by_elem_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_elem_by_elem_cuda_c99 )

    target_link_libraries( test_track_job_track_elem_by_elem_cuda_c99
                          ${SIXTRL_TEST_LIBS} )

    add_test( C99_CudaTrackJobTrackElemByElemTests
              test_track_job_track_elem_by_elem_cuda_c99 )

endif()

if( UNIT_TEST_TARGETS )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY
        POSITION_INDEPENDENT_CODE ON )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY LINKER_LANGUAGE C )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY
        CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY
        CUDA_SEPARABLE_COMPILATION  ON  )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF )

    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CUDA_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY
        CUDA_STANDARD_REQUIRED 11 )

    set_property( TARGET ${UNIT_TEST_TARGETS} APPEND PROPERTY
        COMPILE_OPTIONS ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )
endif()

# end: tests/sixtracklib/cuda/track/CMakeLists.txt
