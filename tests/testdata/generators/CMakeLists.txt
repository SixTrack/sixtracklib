# tests/testdata/generators/CMakeLists.txt

message( STATUS "---- processing tests/testdata_generators/CMakeLists.txt" )

set( GENERATOR_TARGETS )
set( GENERATOR_SOURCES_TO_INSTALL )

set( GENERATORS_LINK_LIBRARIES sixtrack_test sixtrack m dl )

set( SIXTRACKLIB_TESTDATA_DIR "${CMAKE_SOURCE_DIR}/tests/testdata" )
set( TESTDATA_TARGET_OUTPUT   "{SIXTRACKLIB_TESTDATA_DIR}/testdata.generated" )

# -----------------------------------------------------------------------------
# generate_buffer_generic_obj:

add_executable( generate_buffer_generic_obj generate_buffer_generic_obj.c )
list( APPEND GENERATOR_TARGETS generate_buffer_generic_obj )

# -----------------------------------------------------------------------------
# generate_track_be_drift_data:

add_executable( generate_track_be_drift_data generate_track_be_drift_data.c )
list( APPEND GENERATOR_TARGETS generate_track_be_drift_data )

# -----------------------------------------------------------------------------
# generate_track_be_driftexact_data:

add_executable( generate_track_be_driftexact_data
                generate_track_be_driftexact_data.c )
list( APPEND GENERATOR_TARGETS generate_track_be_driftexact_data )

# =============================================================================
# Add properties for all C99 targets:

if( GENERATOR_TARGETS )

    set(  GENERATOR_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )
    list( APPEND GENERATOR_COMPILE_OPTIONS -Wall )
    list( APPEND GENERATOR_COMPILE_OPTIONS -Werror )
    list( APPEND GENERATOR_COMPILE_OPTIONS -pedantic )

    set_property( TARGET ${GENERATOR_TARGETS}
                  PROPERTY LINK_LIBRARIES ${GENERATORS_LINK_LIBRARIES} )

    set_property( TARGET ${GENERATOR_TARGETS} PROPERTY C_STANDARD 99 )
    set_property( TARGET ${GENERATOR_TARGETS} PROPERTY C_STANDARD_REQUIRED ON )

    set_property( TARGET ${GENERATOR_TARGETS}
                  APPEND PROPERTY COMPILE_OPTIONS
                  ${GENERATOR_COMPILE_OPTIONS} )
endif()

# =============================================================================
# Add Python based testdata generators

set( PYTHON_GENERATOR_TARGETS )

if( SIXTRACKL_ENABLE_PYTHON AND PYTHONINTERP_FOUND )

    # prepare the testdata.py file used by the generators to understand the
    # output directory

    set( TESTDATA_PY_IN_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/testdata.py.template" )
    set( TESTDATA_PY_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/testdata.py" )

    configure_file( ${TESTDATA_PY_IN_PATH} ${TESTDATA_PY_OUT_PATH} @ONLY )

    # ------------------------------------------------------------------------
    # generate_track_bbsimple_data.py:

    list( APPEND PYTHON_GENERATOR_TARGETS generate_track_bbsimple_data.py )

    # ------------------------------------------------------------------------
    # generate_track_beambeam_data.py:

    list( APPEND PYTHON_GENERATOR_TARGETS generate_track_beambeam_data.py )

    # ------------------------------------------------------------------------
    # generate_track_lhc_no_bb_data.py:

    list( APPEND PYTHON_GENERATOR_TARGETS generate_track_lhc_no_bb_data.py )

endif()

# =============================================================================
# Create custom command to call all generators:

add_custom_command(
    OUTPUT  ${TESTDATA_TARGET_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E echo "1" >
        ${SIXTRACKLIB_TESTDATA_DIR}/testdata.generated
    COMMENT "Generate testdata ... "
    VERBATIM
)

foreach( GENERATOR_CMD ${GENERATOR_TARGETS} )
    add_custom_command( OUTPUT  ${TESTDATA_TARGET_OUTPUT}
                        APPEND  COMMAND ${GENERATOR_CMD} )
endforeach()

if( PYTHON_GENERATOR_TARGETS )
    foreach( PY_GENERATOR_SCRIPT ${PYTHON_GENERATOR_TARGETS} )
        set( PY_GENERATOR_SCRIPT
             "${CMAKE_CURRENT_SOURCE_DIR}/${PY_GENERATOR_SCRIPT}" )

        add_custom_command(
            OUTPUT ${TESTDATA_TARGET_OUTPUT}
            APPEND COMMAND ${PYTHON_EXECUTABLE} ${PY_GENERATOR_SCRIPT} )

    endforeach()
endif()

add_custom_target( testdata DEPENDS ${TESTDATA_TARGET_OUTPUT} )

# =============================================================================
# Install:

# Note the generators are used in the context of the unit-tests
# therefore, the generators are also part of the CMakeLists.txt generated
# in the tests directory of the installation part
# Here, only the installation of the generator files themselves are handled

if( SIXTRACKL_INSTALL_PROGRAM_TESTS )
    message( STATUS "------ Preparing installation of tests/testdata_generators" )
endif()

if( SIXTRACKL_INSTALL_PROGRAM_TESTS AND GENERATOR_TARGETS )
    set( GENERATED_CMAKELISTS_PROGRAMS )

    string( APPEND GENERATED_CMAKELISTS_PROGRAMS
            "set( GENERATOR_TARGETS )\r\n" )

    string( APPEND GENERATED_CMAKELISTS_PROGRAMS
            "set( GENERATOR_LINK_LIBRARIES \r\n"
            "     SixTracklib::sixtrack_test SixTracklib::sixtrack )\r\n\r\n" )

    foreach( GENERATOR_PRG ${GENERATOR_TARGETS} )
        get_property( GENERATOR_PRG_SOURCES TARGET ${GENERATOR_PRG} PROPERTY SOURCES )
        list( APPEND GENERATOR_SOURCES_TO_INSTALL ${GENERATOR_PRG_SOURCES} )

        string( APPEND GENERATED_CMAKELISTS_PROGRAMS
                "# ${GENERATOR_PRG}: \r\n"
                "add_executable( ${GENERATOR_PRG} ${GENERATOR_PRG_SOURCES} )\r\n"
                "list( APPEND GENERATOR_TARGETS ${GENERATOR_PRG} )\r\n\r\n" )

    endforeach()

    string( APPEND GENERATED_CMAKELISTS_PROGRAMS
            "set( GENERATOR_COMPILE_OPTIONS ${SIXTRACKLIB_CPU_FLAGS} )\r\n"
            "list( APPEND GENERATOR_COMPILE_OPTIONS -Wall )\r\n"
            "list( APPEND GENERATOR_COMPILE_OPTIONS -Werror )\r\n"
            "list( APPEND GENERATOR_COMPILE_OPTIONS -pedantic )"
            "\r\n\r\n"
            "set_property( TARGET \\\${GENERATOR_TARGETS} \r\n"
            "              PROPERTY LINKER_LANGUAGE C )\r\n"
            "set_property( TARGET \\\${GENERATOR_TARGETS} \r\n"
            "              PROPERTY C_STANDARD 99 )\r\n"
            "set_property( TARGET \\\${GENERATOR_TARGETS} \r\n"
            "              PROPERTY C_STANDARD_REQUIRED ON )\r\n"
            "set_property( TARGET \\\${GENERATOR_TARGETS} \r\n"
            "              APPEND PROPERTY COMPILE_OPTIONS "
                           "\\\${GENERATOR_COMPILE_OPTIONS} )\r\n"
            "set_property( TARGET \\\${GENERATOR_TARGETS} \r\n"
            "              PROPERTY LINK_LIBRARIES "
                           "\\\${GENERATOR_LINK_LIBRARIES} )\r\n" )

    # --------------------------------------------------------------------------
    # Install example program sources:

    set( GENERATOR_INSTALL_PREFIX
         "${CMAKE_INSTALL_PREFIX}/tests/testdata_generators" )

    list( REMOVE_DUPLICATES GENERATOR_SOURCES_TO_INSTALL )

    if( GENERATOR_SOURCES_TO_INSTALL )
        install( FILES ${GENERATOR_SOURCES_TO_INSTALL}
                 DESTINATION ${GENERATOR_INSTALL_PREFIX} )

    endif()

    # --------------------------------------------------------------------------
    # Install CMakeLists.txt file for the testdata_generators

    set( CMAKELISTS_TXT_IN_PATH
         "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.install" )

    set( CMAKELISTS_TXT_INSTALL_PATH
         "${GENERATOR_INSTALL_PREFIX}/CMakeLists.txt" )

    install( CODE "set( GENERATED_CMAKELISTS_PROGRAMS
                    \"${GENERATED_CMAKELISTS_PROGRAMS}\" )
        message( STATUS \"Generating: ${CMAKELISTS_TXT_INSTALL_PATH}\" )
        configure_file( \"${CMAKELISTS_TXT_IN_PATH}\"
                        \"${CMAKELISTS_TXT_INSTALL_PATH}\" @ONLY )" )

endif()

#end: tests/testdata/generators/CMakeLists.txt






