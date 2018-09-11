# sixtracklib/tests/sixtracklib/testlib/CMakeLists.txt

# ------------------------------------------------------------------------------
# generate the testdata_files.h header to be used with examples and unit-tests

set( SIXTRACKLIB_TESTLIB_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/testlib" )

set( SIXTRACKLIB_TESTDATA_DIR "${CMAKE_SOURCE_DIR}/tests/testdata" )

set( SIXTRACKLIB_TESTS_TESTDATA_FILES_H_IN_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/testdata_files.h.template" )

set( SIXTRACKLIB_TESTS_TESTDATA_FILES_H_OUT_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}/testdata_files.h" )

set( SIXTRACKLIB_TESTS_TESTDATA_FILES_H_INSTALL_PATH
     "${SIXTRACKLIB_TESTLIB_INSTALL_PATH}/testdata_files.h" )

configure_file( ${SIXTRACKLIB_TESTS_TESTDATA_FILES_H_IN_PATH}
                ${SIXTRACKLIB_TESTS_TESTDATA_FILES_H_OUT_PATH}
                @ONLY )

install( CODE "
    set( SIXTRACKLIB_TESTDATA_DIR \"${CMAKE_INSTALL_PREFIX}/tests/testdata\" )
    message( STATUS \"Generating: ${SIXTRACKLIB_TESTS_TESTDATA_FILES_H_INSTALL_PATH}\" )
    configure_file( \"${SIXTRACKLIB_TESTS_TESTDATA_FILES_H_IN_PATH}\"
                    \"${SIXTRACKLIB_TESTS_TESTDATA_FILES_H_INSTALL_PATH}\" @ONLY ) " )

# -----------------------------------------------------------------------------
# build sixtrack_test library:

set( SIXTRACKL_TESTLIB_TOP_LEVEL_HEADERS
     ../testlib.h )

set( SIXTRACKL_TESTLIB_HEADERS
     generic_buffer_obj.h
     gpu_kernel_tools.h
     random.h
     time_tools.h
     test_particles_tools.h
     test_beam_elements_tools.h
     track_testdata_tools.h
#      test_track_tools.h
)

set( SIXTRACKL_TESTLIB_SOURCES
     details/gpu_kernel_tools.c
     details/random.c
     details/time_tools.c
     details/test_particles_tools.c
     details/test_beam_elements_tools.c
     details/track_testdata_tools.c
#      details/test_track_tools.c
)

add_library( sixtrack_test STATIC
             ${SIXTRACKL_TESTLIB_TOP_LEVEL_HEADERS}
             ${SIXTRACKL_TESTLIB_HEADERS}
             ${SIXTRACKL_TESTLIB_SOURCES}
)

target_include_directories(
    sixtrack_test
    PUBLIC $<INSTALL_INTERFACE:include>
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

target_link_libraries( sixtrack_test sixtrack )

set_target_properties( sixtrack_test PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_test PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_test PROPERTIES DEBUG_POSTFIX d )

target_compile_definitions( sixtrack_test
                            PUBLIC ${SIXTRACKLIB_NAMESPACE_FLAGS} )

target_compile_options( sixtrack_test
                        PRIVATE -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )


# ------------------------------------------------------------------------------
# install :

set( SIXTRACKL_TESTLIB_TOP_LEVEL_INSTALL_INCLUDE_PATH include/sixtracklib )

install( FILES ${SIXTRACKL_TESTLIB_TOP_LEVEL_HEADERS}
         DESTINATION ${SIXTRACKL_TESTLIB_TOP_LEVEL_INSTALL_INCLUDE_PATH} )

set( SIXTRACKL_TESTLIB_INSTALL_CONDIG_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake )

install( TARGETS sixtrack_test
         EXPORT  SixTrackTestlib-targets
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib )

install( EXPORT SixTrackTestlib-targets
         FILE   SixTrackTestlibTargets.cmake
         NAMESPACE SixTracklib::
         DESTINATION ${SIXTRACKL_TESTLIB_INSTALL_CONDIG_PATH}
)

set( SIXTRACKL_TESTLIB_INSTALL_INCLUDE_PATH include/sixtracklib/testlib )

install( FILES ${SIXTRACKL_TESTLIB_HEADERS}
         DESTINATION ${SIXTRACKL_TESTLIB_INSTALL_INCLUDE_PATH} )

#end sixtracklib/common/CMakeLists.txt
