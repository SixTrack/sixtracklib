find_path( Gmp_INCLUDES
  NAMES gmp.h
  PATHS $ENV{GMP_ROOT}
  ${INCLUDE_INSTALL_DIR}
)

find_library( 
    Gmp_LIBRARIES gmp 
    PATHS $ENV{GMP_ROOT} 
    ${LIB_INSTALL_DIR}
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    Gmp
    FOUND_VAR Gmp_FOUND 
    REQUIRED_VARS Gmp_INCLUDES Gmp_LIBRARIES
)

mark_as_advanced( Gmp_INCLUDES Gmp_LIBRARIES )


