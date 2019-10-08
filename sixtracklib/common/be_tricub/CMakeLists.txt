set( SIXTRL_COMMON_BE_TRICUB_SOURCES be_tricub.c )
set( SIXTRL_COMMON_BE_TRICUB_HEADERS be_tricub.h track.h )

add_library( sixtrack_common_be_tricub OBJECT
    ${SIXTRL_COMMON_BE_TRICUB_HEADERS} ${SIXTRL_COMMON_BE_TRICUB_SOURCES}
)

target_include_directories( sixtrack_common_be_tricub PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_be_tricub PROPERTIES
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON C_STANDARD 99
    C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_be_tricub PRIVATE
                        ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_be_tricub as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_be_tricub> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_BE_TRICUB_C99_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/common/be_tricub" )

install( FILES ${SIXTRL_COMMON_BE_TRICUB_HEADERS}
         DESTINATION ${SIXTRACKLIB_COMMON_BE_TRICUB_C99_INSTALL_PATH} )

