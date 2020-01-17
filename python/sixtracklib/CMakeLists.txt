# python/sixtracklib/CMakeLists.txt

# ------------------------------------------------------------------------------
# Generate config.py

set( PYTHON_SIXTRACKLIB_SHARED_LIB  ${PY_SIXTRACKLIB_SHARED_LIB}  )

string( REPLACE ";" "\r\n" PYTHON_SIXTRACKLIB_MODULES_STR
        "${PY_SIXTRACKLIB_MODULES_STR}" )

set( PYTHON_SIXTRL_C99_NAMESPACE_STR ${PY_SIXTRL_C99_NAMESPACE_STR} )
set( PYTHON_SIXTRL_CXX_NAMESPACE_STR ${PY_SIXTRL_CXX_NAMESPACE_STR} )

set( PYTHON_SIXTRL_C99_NAMESPACE_PREFIX_STR
     ${PY_SIXTRL_C99_NAMESPACE_PREFIX_STR} )

set( PYTHON_SIXTRL_CXX_NAMESPACE_PREFIX_STR
     ${PY_SIXTRL_CXX_NAMESPACE_PREFIX_STR} )

set( PYTHON_SIXTRL_ENABLE_APERTURE_CHECK ${PY_SIXTRL_ENABLE_APERTURE_CHECK} )
set( PYTHON_SIXTRL_APERTURE_X_LIMIT ${PY_SIXTRL_APERTURE_X_LIMIT} )
set( PYTHON_SIXTRL_APERTURE_Y_LIMIT ${PY_SIXTRL_APERTURE_Y_LIMIT} )

set( SIXTRL_PY_IN_FILE  config.py.template )
set( SIXTRL_PY_OUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/config.py" )

configure_file( ${SIXTRL_PY_IN_FILE} ${SIXTRL_PY_OUT_FILE} @ONLY )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Install

set( SIXTRL_PY_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/python/sixtracklib/config.py" )

install( CODE
    "${SIXTRL_TRACK_FEATURES_INSTALL_STR}
     ${SIXTRL_APERTURE_CHECK_FEATURES_INSTALL_STR}

     message( STATUS \"Generating: ${SIXTRL_PY_INSTALL_PATH}\" )
     configure_file( \"${SIXTRL_PY_IN_FILE}\" \"${SIXTRL_PY_INSTALL_PATH}\"
     @ONLY ) "
)

# end: python/sixtracklib/CMakeLists.txt
