# sixtracklib/common/tests/CMakeLists.txt 

if( GTEST_FOUND )

    include_directories( ${SIXTRACKL_GTEST_INCLUDE_DIRS} )

    set( UNIT_TEST_TARGETS ) 
    set( SIXTRACKL_TEST_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} 
         ${SIXTRACKL_GTEST_LIBRARIES} )
    
    # --------------------------------------------------------------------------
    # test_track:
    
    add_executable( test_track_cuda test_track.cpp )
    set( UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_track_cuda )
    #add_test( CommonTrackTests test_track_cuda )
    
    # *************************************************************************
    # Set all properties: 
    
    set_property( TARGET ${UNIT_TEST_TARGETS} 
                  PROPERTY LINK_LIBRARIES ${SIXTRACKL_TEST_LIBRARIES} )
                  
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD 11 )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY CXX_STANDARD_REQUIRED ON )
    set_property( TARGET ${UNIT_TEST_TARGETS} PROPERTY COMPILE_OPTIONS 
                  ${SIXTRACKLIB_CPU_FLAGS} -Wall -Werror -pedantic )
    
endif()
