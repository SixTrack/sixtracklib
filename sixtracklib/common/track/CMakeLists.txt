# ----------------------------------------------------------------------------
# sixtracklib/common/track/CMakeLists.txt

set( SIXTRL_COMMON_TRACK_C99_HEADERS
     definitions.h
     track.h
     track_kernel_impl.h
     track_kernel_opt_impl.h
     track_job_base.h
     track_job_cpu.h
)

set( SIXTRL_COMMON_TRACK_C99_SOURCES
     track.c
)

add_library( sixtrack_common_track OBJECT
    ${SIXTRL_COMMON_TRACK_C99_HEADERS} ${SIXTRL_COMMON_TRACK_C99_SOURCES} )

target_include_directories( sixtrack_common_track PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

if( SIXTRACKL_ENABLE_CUDA )
    target_include_directories( sixtrack_common_track
        PUBLIC $<BUILD_INTERFACE:${SIXTRACKL_CUDA_INCLUDE_DIRS}> )
endif()

set_target_properties( sixtrack_common_track PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE C
    C_STANDARD 99 C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_track PRIVATE
    ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

set( SIXTRL_COMMON_TRACK_CXX_HEADERS
     track_job_base.hpp
     track_job_ctrl_arg_base.hpp
     track_job_nodectrl_arg_base.hpp
     track_job_cpu.hpp
)

set( SIXTRL_COMMON_TRACK_CXX_SOURCES
     track_job_base.cpp
     track_job_base_c99.cpp
     track_job_ctrl_arg_base.cpp
     track_job_nodectrl_arg_base.cpp
     track_job_cpu.cpp
     track_job_cpu_c99.cpp
)

add_library( sixtrack_common_track_cxx OBJECT
    ${SIXTRL_COMMON_TRACK_CXX_HEADERS} ${SIXTRL_COMMON_TRACK_CXX_SOURCES}
    ${SIXTRL_COMMON_TRACK_C99_HEADERS}
)

target_include_directories( sixtrack_common_track_cxx PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}> $<INSTALL_INTERFACE:include> )

if( SIXTRACKL_ENABLE_CUDA )
    target_include_directories( sixtrack_common_track_cxx
        PUBLIC $<BUILD_INTERFACE:${SIXTRACKL_CUDA_INCLUDE_DIRS}> )
endif()

set_target_properties( sixtrack_common_track_cxx PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE C
    CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_common_track_cxx PRIVATE
    ${SIXTRACKLIB_CXX_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_common_track and sixtrack_common_track_cxx
# as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_common_track>
    $<TARGET_OBJECTS:sixtrack_common_track_cxx>
    CACHE INTERNAL "" FORCE )

# -----------------------------------------------------------------------------
# install :

if( SIXTRL_COMMON_TRACK_C99_HEADERS )
    install( FILES ${SIXTRL_COMMON_TRACK_C99_HEADERS}
             DESTINATION "${SIXTRL_COMMON_C99_INSTALL_PREFIX}/track" )
endif()

if( SIXTRACKL_ENABLE_CXX AND SIXTRL_COMMON_TRACK_CXX_HEADERS )
    install( FILES ${SIXTRL_COMMON_TRACK_CXX_HEADERS}
             DESTINATION "${SIXTRL_COMMON_CXX_INSTALL_PREFIX}/track" )
endif()

#end: sixtracklib/common/track/CMakeLists.txt
