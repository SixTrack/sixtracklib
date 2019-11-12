# examples/cxx/CMakeLists.txt

message( STATUS "---- processing examples/cxx/CMakeLists.txt" )

set( EXAMPLE_TARGETS )
set( EXAMPLE_CUDA_TARGETS )
set( EXAMPLE_OPENCL_TARGETS )

set( EXAMPLE_SOURCES_TO_INSTALL )

set( EXAMPLE_LINK_LIBRARIES sixtrack_test sixtrack m dl )

# -----------------------------------------------------------------------------
# simple_drift_buffer_cxx:

add_executable( simple_drift_buffer_cxx simple_drift_buffer.cpp )
list( APPEND EXAMPLE_TARGETS simple_drift_buffer_cxx )

# -----------------------------------------------------------------------------
# track_job_cpu_cxx:

add_executable( track_job_cpu_cxx track_job_cpu.cpp )
list( APPEND EXAMPLE_TARGETS track_job_cpu_cxx )

# -----------------------------------------------------------------------------
# track_lhc_no_bb_cxx:

add_executable( track_lhc_no_bb_cxx track_lhc_no_bb.cpp )
list( APPEND EXAMPLE_TARGETS track_lhc_no_bb_cxx )

if( SIXTRACKL_ENABLE_OPENCL )
    # -------------------------------------------------------------------------
    # track_job_cpu_cxx:

    add_executable( track_job_cl_cxx track_job_cl.cpp )
    list( APPEND EXAMPLE_TARGETS track_job_cl_cxx )

    # -------------------------------------------------------------------------
    # run_opencl_kernel_cxx:
    add_executable( run_opencl_kernel_cxx run_opencl_kernel.cpp )
    list( APPEND EXAMPLE_OPENCL_TARGETS run_opencl_kernel_cxx )

    # -------------------------------------------------------------------------
    # track_lhc_no_bb_opencl_cxx:
    add_executable( track_lhc_no_bb_opencl_cxx track_lhc_no_bb_opencl.cpp )
    list( APPEND EXAMPLE_OPENCL_TARGETS track_lhc_no_bb_opencl_cxx )


endif()

# ==============================================================================
# Common settings for all programs:

set( ALL_EXAMPLE_TARGETS
     ${EXAMPLE_TARGETS} ${EXAMPLE_OPENCL_TARGETS} ${EXAMPLE_CUDA_TARGETS} )

if( ALL_EXAMPLE_TARGETS )

    set(  EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND EXAMPLE_COMPILE_OPTIONS -Wall )
    list( APPEND EXAMPLE_COMPILE_OPTIONS -Werror )
    list( APPEND EXAMPLE_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${ALL_EXAMPLE_TARGETS} PROPERTY LINKER_LANGUAGE CXX )
    set_property( TARGET ${ALL_EXAMPLE_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${ALL_EXAMPLE_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${ALL_EXAMPLE_TARGETS}
                  APPEND PROPERTY COMPILE_OPTIONS ${EXAMPLE_COMPILE_OPTIONS} )

    set_property( TARGET ${ALL_EXAMPLE_TARGETS}
                  PROPERTY LINK_LIBRARIES ${EXAMPLE_LINK_LIBRARIES} )
endif()

# ==============================================================================
# Install :

if( SIXTRACKL_INSTALL_EXAMPLES )
    message( STATUS "------ Preparing installation of examples/cxx" )
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
            "set( EXAMPLE_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )\r\n"
            "list( APPEND EXAMPLE_COMPILE_OPTIONS -Wall )\r\n"
            "list( APPEND EXAMPLE_COMPILE_OPTIONS -Werror )\r\n"
            "list( APPEND EXAMPLE_COMPILE_OPTIONS -pedantic )"
            "\r\n\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY LINKER_LANGUAGE CXX )\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY CXX_STANDARD 11 )\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY CXX_STANDARD_REQUIRED ON )\r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              APPEND PROPERTY COMPILE_OPTIONS "
                          "\\\$\{EXAMPLE_COMPILE_OPTIONS} ) \r\n"
            "set_property( TARGET \\\${EXAMPLE_TARGETS} \r\n"
            "              PROPERTY LINK_LIBRARIES "
                          "\\\${EXAMPLE_LINK_LIBRARIES} )\r\n" )

    # --------------------------------------------------------------------------
    # Install example program sources:

    set( EXAMPLES_CXX_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/examples/cxx" )

    list( REMOVE_DUPLICATES EXAMPLE_SOURCES_TO_INSTALL )

    if( EXAMPLE_SOURCES_TO_INSTALL )
        install( FILES ${EXAMPLE_SOURCES_TO_INSTALL}
                 DESTINATION ${EXAMPLES_CXX_INSTALL_PREFIX} )

    endif()

    # --------------------------------------------------------------------------
    # Install CMakeLists.txt file for the c99 examples

    set( CMAKELISTS_TXT_IN_PATH
         "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.install" )

    set( CMAKELISTS_TXT_INSTALL_PATH
         "${EXAMPLES_CXX_INSTALL_PREFIX}/CMakeLists.txt" )

    install( CODE "set( GENERATED_CMAKELISTS_PROGRAMS
                    \"${GENERATED_CMAKELISTS_PROGRAMS}\" )
        message( STATUS \"Generating: ${CMAKELISTS_TXT_INSTALL_PATH}\" )
        configure_file( \"${CMAKELISTS_TXT_IN_PATH}\"
                        \"${CMAKELISTS_TXT_INSTALL_PATH}\" ESCAPE_QUOTES @ONLY )" )
endif()

# end: examples/cxx/CMakeLists.txt
