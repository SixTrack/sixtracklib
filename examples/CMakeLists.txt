# sixtracklib/examples/CMakeLists.txt

set( SIXTRACKLIB_LIB_CONFIG_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake" )
set( SIXTRACKLIB_EXAMPLES_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/examples" )

add_subdirectory( c99 )
add_subdirectory( cxx )

# =============================================================================
# Install:

set( CMAKELISTS_TXT_IN_PATH
         "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.install" )

    set( CMAKELISTS_TXT_INSTALL_PATH
         "${CMAKE_INSTALL_PREFIX}/examples/CMakeLists.txt" )

install( CODE
         "set( SIXTRACKLIB_LIB_CONFIG_PATH \"${SIXTRACKLIB_LIB_CONFIG_PATH}\" )
          message( STATUS \"Generating: ${CMAKELISTS_TXT_INSTALL_PATH}\" )
          configure_file( \"${CMAKELISTS_TXT_IN_PATH}\"
                          \"${CMAKELISTS_TXT_INSTALL_PATH}\" @ONLY )" )

# end: examples/CMakeLists.txt
