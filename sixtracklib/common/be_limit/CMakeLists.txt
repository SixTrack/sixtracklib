# =============================================================================
# end: sixtracklib/common/be_limit/CMakeLitst.txt

set( SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS )

set( SIXTRL_COMMON_BE_LIMIT_C99_HEADERS )
set( SIXTRL_COMMON_BE_LIMIT_C99_SOURCES )

add_library( sixtrack_be_limit OBJECT
    ${SIXTRL_COMMON_BE_LIMIT_C99_HEADERS}
    ${SIXTRL_COMMON_BE_LIMIT_CXX_HEADERS}
    ${SIXTRL_COMMON_BE_LIMIT_C99_SOURCES}
)

# ==============================================================================
# Installation:

if( SIXTRL_COMMON_BE_LIMIT_C99_HEADERS )

endif()

#end:
