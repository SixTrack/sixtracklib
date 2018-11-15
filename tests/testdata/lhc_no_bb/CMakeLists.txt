# tests/testdata/lhc_no_bb/CMakeLists.txt

set( TESTDATA_FILES
     dump3.dat
     fort.2
     fort.3
     fort.8
     fort.16
     beam_elements.bin
     particles_dump.bin
     beam_elements_sixtrack.bin
     particles_dump_sixtrack.bin )

# =============================================================================
# Install:

if( TESTDATA_FILES )
    set( TESTDATA_INSTALL_PREFIX
         "${CMAKE_INSTALL_PREFIX}/tests/testdata/lhc_no_bb" )

    install( FILES ${TESTDATA_FILES} DESTINATION ${TESTDATA_INSTALL_PREFIX} )
endif()

# end: tests/testdata/lhc_no_bb/CMakeLists.txt
