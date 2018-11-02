# examples/c99/CMakeLists.txt

message( STATUS "---- processing examples/c99/CMakeLists.txt" )

set( EXAMPLE_TARGETS )
set( EXAMPLE_CUDA_TARGETS )
set( EXAMPLE_OPENCL_TARGETS )

set( EXAMPLE_SOURCES_TO_INSTALL )

set( EXAMPLE_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# -----------------------------------------------------------------------------
# simple_drift_buffer:

add_executable( simple_drift_buffer_c99  simple_drift_buffer.c   )
list( APPEND EXAMPLE_TARGETS             simple_drift_buffer_c99 )

# -----------------------------------------------------------------------------
# simple_particles_buffer:

add_executable( simple_particles_buffer_c99 simple_particles_buffer.c )
list( APPEND EXAMPLE_TARGETS simple_particles_buffer_c99 )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_c99:

add_executable( track_lhc_no_bb_c99 track_lhc_no_bb.c )
list( APPEND EXAMPLE_TARGETS track_lhc_no_bb_c99 )

# -----------------------------------------------------------------------------
# track_lhc_bb_c99:

add_executable( track_lhc_bb_c99 track_lhc_bb.c )
list( APPEND EXAMPLE_TARGETS track_lhc_bb_c99 )

# -----------------------------------------------------------------------------
# track_bbsimple_c99:

add_executable( track_bbsimple_c99 track_bbsimple.c )
list( APPEND EXAMPLE_TARGETS track_bbsimple_c99 )

# -----------------------------------------------------------------------------
# track_beambeam_c99:

add_executable( track_beambeam_c99 track_beambeam.c )
list( APPEND EXAMPLE_TARGETS track_beambeam_c99 )


if( SIXTRACKL_ENABLE_CUDA )
    # -------------------------------------------------------------------------
    # track_lhc_no_bb_cuda_c99:

    add_executable( track_lhc_no_bb_cuda_c99 track_lhc_no_bb_cuda.c )
    list( APPEND EXAMPLE_CUDA_TARGETS        track_lhc_no_bb_cuda_c99 )
endif()

if( SIXTRACKL_ENABLE_OPENCL )
    # -----------------------------------------------------------------------------
    # run_opencl_kernel:

    add_executable( run_opencl_kernel_c99 run_opencl_kernel.c )
    list( APPEND EXAMPLE_OPENCL_TARGETS run_opencl_kernel_c99 )
    list( APPEND EXAMPLE_SOURCES_TO_INSTALL run_opencl_kernel.cl )

    # -------------------------------------------------------------------------
    # track_lhc_no_bb_cuda_c99:

    add_executable( track_lhc_no_bb_opencl_c99 track_lhc_no_bb_opencl.c )
    list( APPEND EXAMPLE_OPENCL_TARGETS        track_lhc_no_bb_opencl_c99 )

    # -------------------------------------------------------------------------
    # track_lhc_bb_cuda_c99:

    add_executable( track_lhc_bb_opencl_c99    track_lhc_bb_opencl.c )
    list( APPEND EXAMPLE_OPENCL_TARGETS        track_lhc_bb_opencl_c99 )
endif()

# ==============================================================================
# set properties common to all targets:

set( ALL_EXAMPLE_TARGETS
     ${EXAMPLE_TARGETS} ${EXAMPLE_OPENCL_TARGETS} ${EXAMPLE_CUDA_TARGETS} )

if( ALL_EXAMPLE_TARGETS )
    set(  EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND EXAMPLE_COMPILE_OPTIONS -Wall )
    list( APPEND EXAMPLE_COMPILE_OPTIONS -Werror )
    list( APPEND EXAMPLE_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${ALL_EXAMPLE_TARGETS} PROPERTY LINKER_LANGUAGE C )
    set_property( TARGET ${ALL_EXAMPLE_TARGETS} PROPERTY C_STANDARD 99 )
    set_property( TARGET ${ALL_EXAMPLE_TARGETS} PROPERTY C_STANDARD_REQUIRED ON )
    set_property( TARGET ${ALL_EXAMPLE_TARGETS}
                  APPEND PROPERTY COMPILE_OPTIONS ${EXAMPLE_COMPILE_OPTIONS} )

    set_property( TARGET ${ALL_EXAMPLE_TARGETS} PROPERTY
                  LINK_LIBRARIES ${EXAMPLE_LINK_LIBRARIES} )
endif()

# ==============================================================================
# Install :

if( SIXTRACKL_INSTALL_EXAMPLES )
    message( STATUS "------ Preparing installation of examples/c99" )
endif()

if( SIXTRACKL_INSTALL_EXAMPLES AND ALL_EXAMPLE_TARGETS )

    set( GENERATED_CMAKELISTS_PROGRAMS )

    string( APPEND GENERATED_CMAKELISTS_PROGRAMS
            "set( EXAMPLE_TARGETS )\r\n" )

    string( APPEND GENERATED_CMAKELISTS_PROGRAMS
            "set( EXAMPLE_LINK_LIBRARIES \r\n
                  SixTracklib::sixtrack_test SixTracklib::sixtrack )\r\n\r\n" )

    if( EXAMPLE_TARGETS )

        foreach( EXAMPLE_PRG ${EXAMPLE_TARGETS} )
            get_property( EXAMPLE_PRG_SOURCES TARGET ${EXAMPLE_PRG} PROPERTY SOURCES )
            list( APPEND EXAMPLE_SOURCES_TO_INSTALL ${EXAMPLE_PRG_SOURCES} )

            string( APPEND GENERATED_CMAKELISTS_PROGRAMS
                    "# ${EXAMPLE_PRG}: \r\n"
                    "add_executable( ${EXAMPLE_PRG} ${EXAMPLE_PRG_SOURCES} )\r\n"
                    "list( APPEND EXAMPLE_TARGETS ${EXAMPLE_PRG} )\r\n\r\n" )

        endforeach()
    endif()

    if( EXAMPLE_OPENCL_TARGETS )
        string( APPEND GENERATED_CMAKELISTS_PROGRAMS "# OpenCL: \r\n" )
        string( APPEND GENERATED_CMAKELISTS_PROGRAMS
                "if( SIXTRACKL_ENABLE_OPENCL ) \r\n" )

        foreach( EXAMPLE_PRG ${EXAMPLE_OPENCL_TARGETS} )
            get_property( EXAMPLE_PRG_SOURCES TARGET ${EXAMPLE_PRG} PROPERTY SOURCES )
            list( APPEND EXAMPLE_SOURCES_TO_INSTALL ${EXAMPLE_PRG_SOURCES} )

            string( APPEND GENERATED_CMAKELISTS_PROGRAMS
                    "    # ${EXAMPLE_PRG}: \r\n"
                    "    add_executable( ${EXAMPLE_PRG} ${EXAMPLE_PRG_SOURCES} )\r\n"
                    "    list( APPEND EXAMPLE_TARGETS ${EXAMPLE_PRG} )\r\n\r\n" )
        endforeach()

        string( APPEND GENERATED_CMAKELISTS_PROGRAMS "endif()\r\n" )
    endif()

    if( EXAMPLE_CUDA_TARGETS )
        string( APPEND GENERATED_CMAKELISTS_PROGRAMS "\r\n# CUDA: \r\n" )
        string( APPEND GENERATED_CMAKELISTS_PROGRAMS
                "if( SIXTRACKL_ENABLE_CUDA ) \r\n" )

        foreach( EXAMPLE_PRG ${EXAMPLE_CUDA_TARGETS} )
            get_property( EXAMPLE_PRG_SOURCES TARGET ${EXAMPLE_PRG} PROPERTY SOURCES )
            list( APPEND EXAMPLE_SOURCES_TO_INSTALL ${EXAMPLE_PRG_SOURCES} )

            string( APPEND GENERATED_CMAKELISTS_PROGRAMS
                    "    # ${EXAMPLE_PRG}: \r\n"
                    "    add_executable( ${EXAMPLE_PRG} ${EXAMPLE_PRG_SOURCES} )\r\n"
                    "    list( APPEND EXAMPLE_TARGETS ${EXAMPLE_PRG} )\r\n\r\n" )
        endforeach()

        string( APPEND GENERATED_CMAKELISTS_PROGRAMS "endif()\r\n" )
    endif()

    string( APPEND GENERATED_CMAKELISTS_PROGRAMS
            "set( C99_EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )\r\n"
            "list( APPEND C99_EXAMPLE_COMPILE_OPTIONS -Wall )\r\n"
            "list( APPEND C99_EXAMPLE_COMPILE_OPTIONS -Werror )\r\n"
            "list( APPEND C99_EXAMPLE_COMPILE_OPTIONS -pedantic )\r\n\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY LINKER_LANGUAGE C )\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY C_STANDARD 99 )\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY C_STANDARD_REQUIRED ON )\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              APPEND PROPERTY COMPILE_OPTIONS "
                           "\\\${C99_EXAMPLE_COMPILE_OPTIONS} )\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY LINK_LIBRARIES "
                          "\\\${EXAMPLE_LINK_LIBRARIES} )\r\n" )

    # --------------------------------------------------------------------------
    # Install example program sources:

    set( EXAMPLES_C99_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/examples/c99" )

    list( REMOVE_DUPLICATES EXAMPLE_SOURCES_TO_INSTALL )

    if( EXAMPLE_SOURCES_TO_INSTALL )
        install( FILES ${EXAMPLE_SOURCES_TO_INSTALL}
                 DESTINATION ${EXAMPLES_C99_INSTALL_PREFIX} )

    endif()

    # --------------------------------------------------------------------------
    # Install CMakeLists.txt file for the c99 examples

    set( CMAKELISTS_TXT_IN_PATH
         "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.install" )

    set( CMAKELISTS_TXT_INSTALL_PATH
         "${EXAMPLES_C99_INSTALL_PREFIX}/CMakeLists.txt" )

    install( CODE "set( GENERATED_CMAKELISTS_PROGRAMS
                    \"${GENERATED_CMAKELISTS_PROGRAMS}\" )
        message( STATUS \"Generating: ${CMAKELISTS_TXT_INSTALL_PATH}\" )
        configure_file( \"${CMAKELISTS_TXT_IN_PATH}\"
                        \"${CMAKELISTS_TXT_INSTALL_PATH}\" ESCAPE_QUOTES @ONLY )" )
endif()

# end: examples/c99/CMakeLists.txt
