# tests/testdata/bbsimple/CMakeLists.txt

set( TESTDATA_FILES
     dump3.dat
     fort.2
     fort.3
     line.pkl
     beam_elements.bin
     particles_dump.bin
     beam_elements_sixtrack.bin
     particles_dump_sixtrack.bin )


# =============================================================================
# Install:

if( TESTDATA_FILES )
    set( TESTDATA_INSTALL_PREFIX
         "${CMAKE_INSTALL_PREFIX}/tests/testdata/bbsimple" )

    install( FILES ${TESTDATA_FILES} DESTINATION ${TESTDATA_INSTALL_PREFIX} )
endif()

# end: tests/testdata/bbsimple/CMakeLists.txt
