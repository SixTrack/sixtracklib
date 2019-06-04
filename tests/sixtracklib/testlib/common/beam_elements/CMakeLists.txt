# =============================================================================
# tests/sixtracklib/testlib/common/beam_elements/CMakeLists.txt

set( SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_CXX_HEADERS )
set( SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_HEADERS 
     be_limit_rect.h
     be_limit_ellipse.h
     be_dipedge.h
     beam_elements.h
)

set( SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_SOURCES 
     be_limit_rect.c
     be_limit_ellipse.c
     be_dipedge.c
     beam_elements.c
)

add_library( sixtrack_test_common_belems_c99 OBJECT 
    ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_CXX_HEADERS}
    ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_HEADERS}
    ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_SOURCES}
)

target_include_directories( sixtrack_test_common_belems_c99 PRIVATE
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

set_target_properties( sixtrack_test_common_belems_c99 PROPERTIES 
    LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON C_STANDARD 99 
    C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common_belems_c99 PRIVATE
        ${SIXTRACKLIB_C99_FLAGS} ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_test_common_belems_c99 as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_common_belems_c99> CACHE INTERNAL "" FORCE )

# ==============================================================================
# install:

set( SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_INSTALL_PATH 
     include/sixtracklib/testlib/common/beam_elements )


if( SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_HEADERS )
    install( FILES ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_HEADERS}
        DESTINATION ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_INSTALL_PATH} )
endif()

if( SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_CXX_HEADERS AND 
    SIXTRACKL_ENABLE_CXX )
    
    set( SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_CXX_INSTALL_PATH
       ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_C99_INSTALL_PATH} )
    
    install( FILES ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_CXX_HEADERS}
        DESTINATION ${SIXTRL_TESTLIB_COMMON_BEAM_ELEMENTS_CXX_INSTALL_PATH} )
        
endif()
    
# end: tests/sixtracklib/testlib/common/beam_elements/CMakeLists.txt
