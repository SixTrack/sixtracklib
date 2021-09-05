set( SIXTRL_COMMON_BE_BEAMFIELDS_SOURCES
     be_beamfields.c faddeeva.c dawson_approx.c dawson_coeff_xn96_ntmax25.c
     abq2011_coeff_nf24_tm12.c
     abq2011_coeff_poles_nf24_tm12_nt6.c
     abq2011_cf_q1.c
     abq2011_cf_daw_q1.c
     abq2011_root_q1.c )

set( SIXTRL_COMMON_BE_BEAMFIELDS_HEADERS
     be_beamfields.h track.h faddeeva.h abq2011_coeff.h
     dawson_approx.h dawson_coeff.h gauss_fields.h )

add_library( sixtrack_common_be_beamfields OBJECT
    ${SIXTRL_COMMON_BE_BEAMFIELDS_HEADERS}
    ${SIXTRL_COMMON_BE_BEAMFIELDS_SOURCES} )

target_include_directories( sixtrack_common_be_beamfields PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_beamfields PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON C_STANDARD 99
    C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_beamfields PRIVATE
        ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS}
        ${SIXTRACKL_C99_AUTOVEC_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_beamfields as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_beamfields> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install :

if( SIXTRL_COMMON_BE_BEAMFIELDS_HEADERS )
    install( FILES ${SIXTRL_COMMON_BE_BEAMFIELDS_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/be_beamfields" )
endif()
