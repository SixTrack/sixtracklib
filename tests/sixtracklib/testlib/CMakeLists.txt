# sixtracklib/tests/sixtracklib/testlib/CMakeLists.txt

message( STATUS "------ processing tests/sixtracklib/testlib/CMakeLists.txt" )

set( SIXTRL_TESTLIB_INSTALL_C99_PATH
     "${SIXTRL_TESTLIB_INSTALL_C99_PREFIX}/testlib" )

set( SIXTRL_TESTLIB_INSTALL_CXX_PATH
     "${SIXTRL_TESTLIB_INSTALL_CXX_PREFIX}/testlib" )

set( SIXTRACKL_TESTLIB_LINK_LIBRARIES sixtrack  CACHE INTERNAL "" FORCE )
set( SIXTRACKL_TESTLIB_LIBRARY_MODULES
     "" CACHE INTERNAL "" FORCE )

set( SIXTRACKL_TESTLIB_C99_HEADERS )
set( SIXTRACKL_TESTLIB_CXX_HEADERS )

# -----------------------------------------------------------------------------
# Add submodules :

add_subdirectory( testdata )
add_subdirectory( common   )

if( SIXTRACKL_ENABLE_OPENCL )
    add_subdirectory( opencl )
endif()

if( SIXTRACKL_ENABLE_CUDA )
#     add_subdirectory( cuda )
endif()

# -----------------------------------------------------------------------------
# Handle top-level and general files at this level :

set( SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS )
set( SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS )

# -----------------------------------------------------------------------------
# build testlib from local and submodule contributions:

add_library( sixtrack_test SHARED
             ${SIXTRACKL_TESTLIB_C99_HEADERS}
             ${SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS}
             ${SIXTRACKL_TESTLIB_CXX_HEADERS}
             ${SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS}
             ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
)

target_include_directories( sixtrack_test
    PUBLIC $<INSTALL_INTERFACE:include>
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests> )

target_link_libraries( sixtrack_test PUBLIC ${SIXTRACKL_TESTLIB_LINK_LIBRARIES} )

set_target_properties( sixtrack_test PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_test PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_test PROPERTIES C_STANDARD_REQUIRED ON )
set_target_properties( sixtrack_test PROPERTIES DEBUG_POSTFIX d )

target_compile_options( sixtrack_test
                        PRIVATE -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# provide a copy of the library to the python module if it is defined!

if( SIXTRACKL_ENABLE_PYTHON )

    set( DEST_PATH "${CMAKE_SOURCE_DIR}/python/pysixtracklib_test" )

    if( EXISTS ${DEST_PATH} )

        get_target_property( OUT_LIB_NAME_PREFIX  sixtrack_test PREFIX )
        get_target_property( OUT_LIB_NAME sixtrack_test LIBRARY_OUTPUT_NAME )
        get_target_property( OUT_LIB_NAME_POSTFIX sixtrack_test RELEASE_POSTFIX )

        if(  NOT OUT_LIB_NAME_PREFIX )
            set( OUT_LIB_NAME_PREFIX "lib" )
        endif()


        if(  NOT OUT_LIB_NAME_POSTFIX )
            set( OUT_LIB_NAME_POSTFIX ".so" )
        endif()

        if(  NOT OUT_LIB_NAME )
            set( OUT_LIB_NAME "sixtrack_test" )
        endif()

        set( LIB_NAME
             "${OUT_LIB_NAME_PREFIX}${OUT_LIB_NAME}${OUT_LIB_NAME_POSTFIX}" )

        set( IN_PATH   $<TARGET_FILE:sixtrack_test> )
        set( DEST_PATH "${DEST_PATH}/${LIB_NAME}" )

        message( STATUS "------ python: providing a copy of testlib "
                        "library to ${DEST_PATH}" )

        set( PY_SIXTRACKLIB_TEST_SHARED_LIB ${LIB_NAME}
             CACHE INTERNAL "" FORCE )

        add_custom_command( TARGET sixtrack_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${IN_PATH} ${DEST_PATH} )
    endif()
endif()

# ------------------------------------------------------------------------------
# install :

set( TESTLIB_INSTALL_CONFIG_PATH "${SIXTRL_TESTLIB_INSTALL_LIB_PREFIX}/cmake" )

install( TARGETS sixtrack_test
         EXPORT  SixTrackTestlib-targets
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib )

install( EXPORT SixTrackTestlib-targets
         FILE   SixTrackTestlibTargets.cmake
         NAMESPACE SixTracklib::
         DESTINATION ${TESTLIB_INSTALL_CONFIG_PATH}
)

if( SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS )
    install( FILES ${SIXTRACKL_TESTLIB_LOCAL_C99_HEADERS}
             DESTINATION ${SIXTRL_TESTLIB_INSTALL_C99_PATH} )
endif()

if( ( SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS ) AND ( SIXTRACKL_ENABLE_CXX ) )
    install( FILES ${SIXTRACKL_TESTLIB_LOCAL_CXX_HEADERS}
             DESTINATION ${SIXTRL_TESTLIB_INSTALL_CXX_PATH} )
endif()

#end sixtracklib/common/CMakeLists.txt
