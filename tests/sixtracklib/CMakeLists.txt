# ==============================================================================
# tests/sixtracklib/CMakeLists.txt

# ------------------------------------------------------------------------------
# C Namespace handling:

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    set( SIXTRACKLIB_TESTS_NAMESPACE_FLAGS
         "__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
else()
    set( SIXTRACKLIB_TESTS_NAMESPACE_FLAGS "__NAMESPACE=st_" )
endif()

# ------------------------------------------------------------------------------

set( SIXTRL_TESTLIB_INSTALL_C99_PREFIX
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib" )

set( SIXTRL_TESTLIB_INSTALL_CXX_PREFIX
     ${SIXTRL_TESTLIB_INSTALL_C99_PREFIX} )

set( SIXTRL_TESTLIB_INSTALL_LIB_PREFIX "${CMAKE_INSTALL_PREFIX}/lib" )

# ------------------------------------------------------------------------------
# ---- build the sixtrack_test library

add_subdirectory( testlib )

# ------------------------------------------------------------------------------
# ---- run the submodule - unit-tests

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )

    set( SIXTRACKL_TEST_LIBRARIES sixtrack_test sixtrack )

    add_subdirectory( common )
#     add_custom_target( testdata DEPENDS testdata_common )

    if( SIXTRACKL_ENABLE_OPENCL )
        add_subdirectory( opencl )
    endif()

    if( SIXTRACKL_ENABLE_CUDA )
        add_subdirectory( cuda )
    endif()

endif()

# ==============================================================================
# Install:

install( FILE testlib.h
         DESTINATION "${SIXTRL_TESTLIB_INSTALL_C99_PREFIX}/testlib" )

if( SIXTRACKL_ENABLE_CXX )

    install( FILE testlib.hpp
             DESTINATION "${SIXTRL_TESTLIB_INSTALL_CXX_PREFIX}/testlib" )

endif()

# end: tests/sixtracklib/CMakeLists.txt

