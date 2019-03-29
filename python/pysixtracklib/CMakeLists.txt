# python/pysixtracklib/CMakeLists.txt

# ------------------------------------------------------------------------------
# Generate sixtracklibconf.py

set( PYTHON_SIXTRACKLIB_SHARED_LIB  ${PY_SIXTRACKLIB_SHARED_LIB}  )
set( PYTHON_SIXTRACKLIB_MODULES_STR ${PY_SIXTRACKLIB_MODULES_STR} )

set( SIXTRL_PY_IN_FILE  sixtracklibconf.py.template )
set( SIXTRL_PY_OUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/sixtracklibconf.py" )

configure_file( ${SIXTRL_PY_IN_FILE} ${SIXTRL_PY_OUT_FILE} @ONLY )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Install

set( SIXTRL_PY_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/python/pysixtracklib/sixtracklibconf.py" )

install( CODE
    "set( PYTHON_SIXTRACKLIB_SHARED_LIB  ${PY_SIXTRACKLIB_SHARED_LIB}  )
     set( PYTHON_SIXTRACKLIB_MODULES_STR ${PY_SIXTRACKLIB_MODULES_STR} )
     message( STATUS \"Generating: ${SIXTRL_PY_INSTALL_PATH}\" )
     configure_file( \"${SIXTRL_PY_IN_FILE}\" \"${SIXTRL_PY_INSTALL_PATH}\"
                     @ONLY ) "
)

# end: python/pysixtracklib/CMakeLists.txt
