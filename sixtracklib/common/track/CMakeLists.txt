# ----------------------------------------------------------------------------
# sixtracklib/common/track/CMakeLists.txt

set( SIXTRL_COMMON_TRACK_C99_HEADERS
     definitions.h
     track_job_base.h
)

set( SIXTRL_COMMON_TRACK_C99_SOURCES )

set( SIXTRL_COMMON_TRACK_CXX_HEADERS
     track_job_base.hpp
     track_job_ctrl_arg_base.hpp
     track_job_nodectrl_arg_base.hpp
)

set( SIXTRL_COMMON_TRACK_CXX_SOURCES
     track_job_base.cpp
     track_job_base_c99.cpp
     track_job_ctrl_arg_base.cpp
     track_job_nodectrl_arg_base.cpp
)

add_library( sixtrack_common_track_cxx OBJECT
    ${SIXTRL_COMMON_TRACK_CXX_HEADERS}
    ${SIXTRL_COMMON_TRACK_C99_HEADERS}
    ${SIXTRL_COMMON_TRACK_CXX_SOURCES}
)

target_include_directories( sixtrack_common_track_cxx PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> )

set_target_properties( sixtrack_common_track_cxx PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE C
    CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_track_cxx PRIVATE
    -Wall -Werror -pedantic -ansi ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_internal as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES
    ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_track_cxx> CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------
# install :

set( SIXTRACKLIB_COMMON_TRACK_INSTALL_PATH
     include/sixtracklib/common/track )

if( SIXTRL_COMMON_TRACK_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_TRACK_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_TRACK_INSTALL_PATH} )
endif()

if( SIXTRL_COMMON_TRACK_CXX_HEADERS )
    install( FILES ${SIXTRL_COMMON_TRACK_CXX_HEADERS}
             DESTINATION ${SIXTRACKLIB_COMMON_TRACK_INSTALL_PATH} )
endif()

#end: sixtracklib/common/track/CMakeLists.txt
