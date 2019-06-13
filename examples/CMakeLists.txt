# sixtracklib/examples/CMakeLists.txt

# ==============================================================================
# Propagate these settings to the install sections in the subdirectories

if( SIXTRACKL_INSTALL_EXAMPLES )
    set( SIXTRACKLIB_LIB_CONFIG_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake" )
    set( SIXTRACKLIB_EXAMPLES_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/examples" )
endif()

# ==============================================================================
# Handle special case if OpenCL headers have been downloaded by cmake: 

if( SIXTRACKL_ENABLE_OPENCL AND CONTRIBUTED_CXX_HEADER AND EXT_OCLCXX_DIR AND 
    EXISTS ${CONTRIBUTED_CXX_HEADER} )
    
    include_directories( ${EXT_OCLCXX_DIR} )
    
endif()

# ==============================================================================
# Add subdirectories for the different bindings

add_subdirectory( c99 )

if( SIXTRACKL_ENABLE_CXX )
    add_subdirectory( cxx )
endif()

#if( SIXTRACKL_ENABLE_PYTHON )
#    add_subdirectory( python )
#endif()

# =============================================================================
# Install:

if( SIXTRACKL_INSTALL_EXAMPLES )
    set( EXAMPLES_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/examples" )

    set( CMAKELISTS_TXT_IN_PATH
         "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.install" )

    set( CMAKELISTS_TXT_INSTALL_PATH
         "${SIXTRACKLIB_EXAMPLES_INSTALL_PATH}/CMakeLists.txt" )

    install( CODE
         "set( SIXTRACKLIB_LIB_CONFIG_PATH \"${SIXTRACKLIB_LIB_CONFIG_PATH}\" )
          message( STATUS \"Generating: ${CMAKELISTS_TXT_INSTALL_PATH}\" )
          configure_file( \"${CMAKELISTS_TXT_IN_PATH}\"
                          \"${CMAKELISTS_TXT_INSTALL_PATH}\" @ONLY )" )

    # --------------------------------------------------------------------------
    # Build a settings file based on the build options

    set( SETTINGS_OPTIONS_AVAILABLE_MODULES )

    if( SIXTRACKLIB_SUPPORTED_MODULES )

        list( LENGTH SIXTRACKLIB_SUPPORTED_MODULES NUM_MODULES )
        list( LENGTH SIXTRACKLIB_SUPPORTED_MODULES_VALUES NUM_VALUES )

        if( NOT( NUM_MODULES EQUAL NUM_VALUES ) )
            message( FATAL_ERROR "------ Inconsistency in modules list: Stopping" )
        endif()

        math( EXPR NUM_MODULES "${NUM_MODULES}-1" )

        foreach( ii RANGE ${NUM_MODULES} )
            list( GET SIXTRACKLIB_SUPPORTED_MODULES ${ii} MODULE_NAME )
            list( GET SIXTRACKLIB_SUPPORTED_MODULES_VALUES ${ii} MODULE_VALUE )

            if( MODULE_VALUE STREQUAL "1" )
                string( APPEND SETTINGS_OPTIONS_AVAILABLE_MODULES
                        "option( SIXTRACKL_ENABLE_${MODULE_NAME} \r\n"
                        "        \\\"Enable examples using module ${MODULE_NAME}\\\"\r\n"
                        "        On  )\r\n\r\n" )
            endif()

        endforeach()
    endif()

    set( SETTINGS_OPTIONS_AVAILABLE_BINDINGS )

    if( SIXTRACKLIB_SUPPORTED_BINDINGS )

        list( LENGTH SIXTRACKLIB_SUPPORTED_BINDINGS NUM_BINDINGS )
        list( LENGTH SIXTRACKLIB_SUPPORTED_BINDINGS_VALUES NUM_VALUES )

        if( NOT( NUM_BINDINGS EQUAL NUM_VALUES ) )
            message( FATAL_ERROR "------ Inconsistency in bindings list: Stopping" )
        endif()

        math( EXPR NUM_BINDINGS "${NUM_BINDINGS}-1" )

        foreach( ii RANGE ${NUM_BINDINGS} )
            list( GET SIXTRACKLIB_SUPPORTED_BINDINGS ${ii} BINDINGS_NAME )
            list( GET SIXTRACKLIB_SUPPORTED_BINDINGS_VALUES ${ii} BINDINGS_VALUE )

            if( BINDINGS_VALUE STREQUAL "1" )
                string( APPEND SETTINGS_OPTIONS_AVAILABLE_BINDINGS
                        "option( SIXTRACKL_ENABLE_${BINDINGS_NAME} \r\n"
                        "        \\\"Enable examples using module ${BINDINGS_NAME}\\\"\r\n"
                        "        On  )\r\n\r\n" )
            endif()

        endforeach()
    endif()


    set( EXAMPLES_SETTINGS_IN_PATH
         "${CMAKE_CURRENT_SOURCE_DIR}/Settings.cmake.install" )

    set( EXAMPLES_SETTINGS_INSTALL_PATH
         "${SIXTRACKLIB_EXAMPLES_INSTALL_PATH}/Settings.cmake" )

    install( CODE
        "set( SIXTRACKLIB_LIB_CONFIG_PATH \"${SIXTRACKLIB_LIB_CONFIG_PATH}\" )
         set( SETTINGS_OPTIONS_AVAILABLE_MODULES  \"${SETTINGS_OPTIONS_AVAILABLE_MODULES}\" )
         set( SETTINGS_OPTIONS_AVAILABLE_BINDINGS \"${SETTINGS_OPTIONS_AVAILABLE_BINDINGS}\" )
         message( STATUS \"Generating: ${EXAMPLES_SETTINGS_INSTALL_PATH}\" )
         configure_file( \"${EXAMPLES_SETTINGS_IN_PATH}\"
                         \"${EXAMPLES_SETTINGS_INSTALL_PATH}\" @ONLY )" )
endif()

# end: examples/CMakeLists.txt
