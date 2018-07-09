# sixtracklib/tests/sixtracklib/CMakeLists.txt

# ------------------------------------------------------------------------------
# C Namespace handling:

if( SIXTRACKL_C_NAMESPACE_PREFIX )
    set( SIXTRACKLIB_TESTS_NAMESPACE_FLAGS
         "__NAMESPACE=${SIXTRACKL_C_NAMESPACE_PREFIX}" )
else()
    set( SIXTRACKLIB_TESTS_NAMESPACE_FLAGS "__NAMESPACE=st_" )
endif()

# ------------------------------------------------------------------------------
# ---- build the sixtrack_test library

add_subdirectory( testlib )

# ------------------------------------------------------------------------------
# ---- run the submodule - unit-tests

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )

    set( SIXTRACKL_TEST_LIBRARIES sixtrack_test sixtrack )

    add_subdirectory( common )
    add_custom_target( testdata DEPENDS testdata_common )

    if( SIXTRACKL_ENABLE_OPENCL )
        add_subdirectory( opencl )
    endif()

    if( SIXTRACKL_ENABLE_CUDA )
        add_subdirectory( cuda )
    endif()

endif()

