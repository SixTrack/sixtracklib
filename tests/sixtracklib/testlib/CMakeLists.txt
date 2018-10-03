# sixtracklib/tests/sixtracklib/testlib/CMakeLists.txt

set( SIXTRACKL_TESTLIB_LINK_LIBRARIES sixtrack  CACHE INTERNAL "" FORCE )

if( GTEST_FOUND )
    set(  SIXTRACKL_TESTLIB_LINK_LIBRARIES
        ${SIXTRACKL_TESTLIB_LINK_LIBRARIES}
        ${SIXTRACKL_GTEST_LIBRARIES} CACHE INTERNAL "" FORCE )
endif()

set( SIXTRACKL_TESTLIB_LIBRARY_MODULES
     "" CACHE INTERNAL "" FORCE )

set( SIXTRACKL_TESTLIB_C99_HEADERS )
set( SIXTRACKL_TESTLIB_CXX_HEADERS )

# -----------------------------------------------------------------------------
# Add submodules :

add_subdirectory( testdata )
add_subdirectory( common   )

if( SIXTRACKL_ENABLE_OPENCL )
    add_subdirectory( opencl )
endif()

if( SIXTRACKL_ENABLE_CUDA )
    add_subdirectory( cuda )
endif()

# -----------------------------------------------------------------------------
# Handle top-level and general files at this level :

set( SIXTRACKL_TESTLIB_TOP_LEVEL_HEADERS ../testlib.h )
set( SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS )
set( SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS )

# -----------------------------------------------------------------------------
# build testlib from local and submodule contributions:

add_library( sixtrack_test SHARED
             ${SIXTRACKL_TESTLIB_C99_HEADERS}
             ${SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS}
             ${SIXTRACKL_TESTLIB_CXX_HEADERS}
             ${SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS}
             ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
)

target_include_directories( sixtrack_test
    PUBLIC $<INSTALL_INTERFACE:include>
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests>
    PUBLIC $<BUILD_INTERFACE:${SIXTRACKL_GTEST_INCLUDE_DIRS}> )

target_link_libraries( sixtrack_test PUBLIC ${SIXTRACKL_TESTLIB_LINK_LIBRARIES} )

set_target_properties( sixtrack_test PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_test PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_test PROPERTIES C_STANDARD_REQUIRED ON )
set_target_properties( sixtrack_test PROPERTIES DEBUG_POSTFIX d )

target_compile_options( sixtrack_test
                        PRIVATE -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# install :

set( TESTLIB_INSTALL_CONFIG_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake )

install( TARGETS sixtrack_test
         EXPORT  SixTrackTestlib-targets
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib )

install( EXPORT SixTrackTestlib-targets
         FILE   SixTrackTestlibTargets.cmake
         NAMESPACE SixTracklib::
         DESTINATION ${TESTLIB_INSTALL_CONFIG_PATH}
)

if( SIXTRACKL_TESTLIB_TOP_LEVEL_HEADERS )
    set( TESTLIB_TOP_LEVEL_INSTALL_INCLUDE_PATH include/sixtracklib )

    install( FILES ${SIXTRACKL_TESTLIB_TOP_LEVEL_HEADERS}
             DESTINATION ${TESTLIB_TOP_LEVEL_INSTALL_INCLUDE_PATH} )
endif()

if( SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS )
    set( TESTLIB_C99_INSTALL_PATH include/sixtracklib/testlib )

    install( FILES ${SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS}
             DESTINATION ${TESTLIB_C99_INSTALL_PATH} )
endif()

if( SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS )
    set( TESTLIB_CXX_INSTALL_PATH include/sixtracklib/testlib )

    install( FILES ${SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS}
             DESTINATION ${TESTLIB_CXX_INSTALL_PATH} )
endif()

#end sixtracklib/common/CMakeLists.txt
