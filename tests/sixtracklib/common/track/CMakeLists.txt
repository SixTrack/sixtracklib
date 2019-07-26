# ==============================================================================
# sixtracklib/tests/sixtracklib/common/track/CMakeLists.txt

if( GTEST_FOUND )
    set( UNIT_TEST_TARGETS )
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES}
         ${SIXTRACKL_GTEST_LIBRARIES} )

    # ==========================================================================
    # test_track_job_setup_cpu_cxx:

    add_executable( test_track_job_setup_cpu_cxx test_track_job_setup_cxx.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_setup_cpu_cxx )
    add_test( CXX_Cpu_CpuTrackJobSetupTests test_track_job_setup_cpu_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_setup_cpu_c99:

    add_executable( test_track_job_setup_cpu_c99 test_track_job_setup_c99.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_job_setup_cpu_c99 )
    add_test( C99_Cpu_CpuTrackJobSetupTests test_track_job_setup_cpu_c99 )

    # ==========================================================================
    # test_track_job_fetch_particle_addr_cpu_cxx:

    add_executable( test_track_job_fetch_particle_addr_cpu_cxx
                    test_track_job_fetch_particle_addr_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_fetch_particle_addr_cpu_cxx )

    add_test( CXX_Cpu_CpuTrackJobFetchParticleAddrTests
              test_track_job_fetch_particle_addr_cpu_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_fetch_particle_addr_cpu_c99:

    add_executable( test_track_job_fetch_particle_addr_cpu_c99
                    test_track_job_fetch_particle_addr_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_fetch_particle_addr_cpu_c99 )

    add_test( C99_Cpu_CpuTrackJobFetchParticleAddrTests
              test_track_job_fetch_particle_addr_cpu_c99 )

    # ==========================================================================
    # test_track_job_track_until_cpu_cxx:

    add_executable( test_track_job_track_until_cpu_cxx
                    test_track_job_track_until_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_until_cpu_cxx )

    add_test( CXX_Cpu_CpuTrackJobTrackUntilTests
        test_track_job_track_until_cpu_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_track_until_cpu_c99:

    add_executable( test_track_job_track_until_cpu_c99
                    test_track_job_track_until_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_until_cpu_c99 )

    add_test( C99_Cpu_CpuTrackJobTrackUntilTests
              test_track_job_track_until_cpu_c99 )

    # ==========================================================================
    # test_track_job_track_line_cpu_cxx:

    add_executable( test_track_job_track_line_cpu_cxx
                    test_track_job_track_line_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_line_cpu_cxx )

    add_test( CXX_Cpu_CpuTrackJobTrackLineTests
              test_track_job_track_line_cpu_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_track_line_cpu_c99:

    add_executable( test_track_job_track_line_cpu_c99
                    test_track_job_track_line_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_line_cpu_c99 )

    add_test( C99_Cpu_CpuTrackJobTrackLineTests
              test_track_job_track_line_cpu_c99 )

    # ==========================================================================
    # test_track_job_track_elem_by_elem_cpu_cxx:

    add_executable( test_track_job_track_elem_by_elem_cpu_cxx
                    test_track_job_track_elem_by_elem_cxx.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_elem_by_elem_cpu_cxx )

    add_test( CXX_Cpu_CpuTrackJobTrackElemByElemTests
              test_track_job_track_elem_by_elem_cpu_cxx )

    # --------------------------------------------------------------------------
    # test_track_job_track_elem_by_elem_cpu_c99:

    add_executable( test_track_job_track_elem_by_elem_cpu_c99
                    test_track_job_track_elem_by_elem_c99.cpp )

    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS}
         test_track_job_track_elem_by_elem_cpu_c99 )

    add_test( C99_Cpu_CpuTrackJobTrackElemByElemTests
              test_track_job_track_elem_by_elem_cpu_c99 )
endif()

if( UNIT_TEST_TARGETS )
        set_property( TARGET ${UNIT_TEST_TARGETS} APPEND PROPERTY
            INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}> )

        set_property( TARGET ${UNIT_TEST_TARGETS}
                      PROPERTY POSITION_INDEPENDENT_CODE ON )

        set_property( TARGET ${UNIT_TEST_TARGETS}
                      PROPERTY LINKER_LANGUAGE C )

        set_property( TARGET ${UNIT_TEST_TARGETS}
                      PROPERTY LINK_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} )

        set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
        set_property( TARGET ${UNIT_TEST_TARGETS}
                      PROPERTY CXX_STANDARD_REQUIRED ON )

        set_property( TARGET ${UNIT_TEST_TARGETS}
                      APPEND PROPERTY COMPILE_OPTIONS
                      ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )
endif()

# end: sixtracklib/tests/sixtracklib/common/track/CMakeLists.txt
